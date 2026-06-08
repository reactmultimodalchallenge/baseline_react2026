import logging
import math
import os
from functools import partial
from pathlib import Path

import hydra
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from framework.modules.post_processor import Processor
from framework.perfrdiff_rewrite_weight import losses as rewrite_losses
from framework.perfrdiff_rewrite_weight.modifier.network import MainNetUnified
from framework.utils.compute_metrics import compute_eeg_metrics, compute_metrics
from framework.utils.util import from_pretrained_checkpoint
from utils.util import AverageMeter

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            resumed_training: bool = False,
            generic: DictConfig = None,
            model: DictConfig = None,
            criterion: DictConfig = None,
            person_specific: DictConfig = None,
            main_model: DictConfig = None,
            pretrained: DictConfig = None,
            batch_size: int = 4,
            post_config_name: str = "configs/shared/model/emotion_autoencoder.yaml",
            post_clip_length: int = 1000,
            data_clamp: bool = True,
            num_eval_preds: int = 10,
            eval_clip_batch_size: int = 8,
            parallel_eval_preds: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.resumed_training = resumed_training
        self.trainer_cfg = generic
        self.model_cfg = model
        self.criterion_cfg = criterion
        self.person_specific_cfg = person_specific
        self.main_model_cfg = main_model
        self.pretrained_cfg = pretrained
        self.batch_size = batch_size
        self.post_config_name = post_config_name
        self.post_clip_length = post_clip_length
        self.data_clamp = data_clamp
        self.num_eval_preds = num_eval_preds
        self.eval_clip_batch_size = eval_clip_batch_size
        self.parallel_eval_preds = parallel_eval_preds
        self.kwargs = kwargs
        self.task = kwargs.get("task", "online")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_eeg = self._as_bool(self.trainer_cfg.get("train_eeg", False))
        self.train_eeg_head_only = self._as_bool(self.trainer_cfg.get("train_eeg_head_only", False))
        self.train_eeg = self.train_eeg or self.train_eeg_head_only
        self.eval_eeg = self._as_bool(self.trainer_cfg.get("eval_eeg", False))

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _resolve_checkpoint_path(path):
        if path is None or str(path).strip() == "":
            return None
        path = str(path)
        if os.path.isabs(path):
            return path
        return hydra.utils.to_absolute_path(path)

    def set_data_module(self, data_module):
        self.data_module = data_module

    def _ensure_eeg_data_enabled(self, stage):
        if stage == "fit" and self.train_eeg:
            if hasattr(self.data_module, "train_set_cfg"):
                self.data_module.train_set_cfg.load_eeg_l = True
            if hasattr(self.data_module, "val_set_cfg"):
                self.data_module.val_set_cfg.load_eeg_l = True
        if stage == "test" and self.eval_eeg and hasattr(self.data_module, "test_set_cfg"):
            self.data_module.test_set_cfg.load_eeg_l = True

    def _rewrite_cfg(self):
        return OmegaConf.create(
            {
                "main_model": OmegaConf.to_container(self.main_model_cfg, resolve=True),
                "person_specific": OmegaConf.to_container(self.person_specific_cfg, resolve=True),
            }
        )

    def _build_diffusion(self, stage):
        model_cfg = self.model_cfg
        if stage == "test" and self.parallel_eval_preds and self.num_eval_preds > 1:
            model_cfg = OmegaConf.create(OmegaConf.to_container(self.model_cfg, resolve=True))
            if model_cfg.diff_model.get("diffusion_prior") is not None:
                model_cfg.diff_model.diffusion_prior.scheduler.num_preds = self.num_eval_preds
            model_cfg.diff_model.diffusion_decoder.scheduler.num_preds = self.num_eval_preds

        model = instantiate(
            model_cfg.diff_model,
            stage=stage,
            resumed_training=False,
            auto_load_ckpt=False,
            latent_embedder=model_cfg.latent_embedder
            if hasattr(model_cfg, "latent_embedder") else None,
            audio_encoder=model_cfg.audio_encoder
            if hasattr(model_cfg, "audio_encoder") else None,
            **self.kwargs,
            _recursive_=False,
        )
        model.to(self.device)
        self._load_pretrained_diffusion(model)
        return model

    def _load_pretrained_diffusion(self, model):
        if self.pretrained_cfg is None:
            raise ValueError("Missing trainer.pretrained configuration for rewrite-weight diffusion.")

        def load_required(path, module, label):
            checkpoint_path = hydra.utils.to_absolute_path(path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Missing pretrained {label} checkpoint: {checkpoint_path}. "
                    "Please place the pretrained diffusion weights under pretrained_models/ "
                    "or override trainer.pretrained.*."
                )
            from_pretrained_checkpoint(checkpoint_path, module, self.device)

        if getattr(model, "diffusion_prior", None) is not None:
            load_required(self.pretrained_cfg.diffusion_prior, model.diffusion_prior.model, "DiffusionPriorNetwork")
        load_required(self.pretrained_cfg.diffusion_decoder, model.diffusion_decoder.model, "TransformerDenoiser")

    def _build_model(self, stage):
        diffusion = self._build_diffusion(stage)
        model = MainNetUnified(self._rewrite_cfg(), diffusion, self.device)
        model.to(self.device)
        return model

    def _build_criterion(self):
        return partial(getattr(rewrite_losses, self.criterion_cfg.type), **self.criterion_cfg.args)

    def _build_optimizer(self, model):
        cfg = self.main_model_cfg.optimizer_hypernet.args
        if self.train_eeg_head_only:
            model.freeze_except_eeg_head()
            params = list(model.eeg_head().parameters())
            trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            logger.info("EEG head-only trainable parameter tensors: %s", trainable_names)
            print(f"EEG head-only trainable parameter tensors: {trainable_names}")
        else:
            params = list(model.modifier_parameters(include_eeg_head=self.train_eeg))
        return optim.SGD(
            params=params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    def _personal_condition_mode(self):
        mode = self.main_model_cfg.args.get("personal_condition_mode", "3dmm_personality")
        if mode == "history_personality":
            return "3dmm_personality"
        return mode

    @staticmethod
    def _infer_checkpoint_personal_condition_mode(checkpoint):
        mode = checkpoint.get("personal_condition_mode")
        if mode == "history_personality":
            return "3dmm_personality"
        if mode is not None:
            return mode

        state_dict = checkpoint.get("state_dict", {})
        if any(key.startswith("personality_fusion.") for key in state_dict):
            return "3dmm_personality"
        if any(key.startswith("personality_encoder.") for key in state_dict):
            return "personality_only"
        return "3dmm_only"

    def _validate_checkpoint_personal_condition_mode(self, checkpoint, checkpoint_path):
        checkpoint_mode = self._infer_checkpoint_personal_condition_mode(checkpoint)
        current_mode = self._personal_condition_mode()
        if checkpoint_mode != current_mode:
            raise ValueError(
                "ModifierNetwork checkpoint personal_condition_mode mismatch. "
                f"Checkpoint: {checkpoint_mode}; current config: {current_mode}; path: {checkpoint_path}. "
                "Use a checkpoint trained with the same mode, or set "
                f"trainer.main_model.args.personal_condition_mode={checkpoint_mode} for evaluation."
            )

    @staticmethod
    def _checkpoint_has_eeg_head(checkpoint):
        state_dict = checkpoint.get("state_dict", {})
        return any(key.startswith("eeg_head.") for key in state_dict)

    def _load_eeg_head_checkpoint(self, model, path):
        checkpoint_path = self._resolve_checkpoint_path(path)
        if checkpoint_path is None:
            return
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Missing EEG head checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(key.startswith("eeg_head.") for key in state_dict):
            state_dict = {
                key[len("eeg_head."):]: value
                for key, value in state_dict.items()
                if key.startswith("eeg_head.")
            }
        if not model.has_eeg_head():
            raise RuntimeError("Cannot load EEG head checkpoint because main_net.eeg_head is disabled.")
        model.eeg_head().load_state_dict(state_dict)
        model.to(self.device)
        logger.info("Loaded EEG head checkpoint: %s", checkpoint_path)

    def _load_modifier_checkpoint_file(self, model, path, require_eeg_head=False):
        checkpoint_path = self._resolve_checkpoint_path(path)
        if checkpoint_path is None:
            raise ValueError("trainer.pretrained.modifier_checkpoint is required for EEG head-only training.")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Missing ModifierNetwork checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._validate_checkpoint_personal_condition_mode(checkpoint, checkpoint_path)
        if require_eeg_head and not self._checkpoint_has_eeg_head(checkpoint):
            raise RuntimeError(f"ModifierNetwork checkpoint has no trained EEG head: {checkpoint_path}")
        model.load_modifier_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        logger.info("Loaded pretrained ModifierNetwork checkpoint: %s", checkpoint_path)

    def _modifier_dir(self, run_key="current_runid"):
        ckpt_root = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir")))
        run_id = self.kwargs.get(run_key) or self.kwargs.get("current_runid")
        ckpt_dir = ckpt_root / str(run_id) / "ModifierNetwork"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def _save_checkpoint(self, model, optimizer, epoch=None, best=False, last=False,
                         save_epoch=False, best_loss=float("inf")):
        ckpt_dir = self._modifier_dir("current_runid")
        def save_modifier_checkpoint(path):
            checkpoint = {
                "epoch": epoch if epoch is not None else None,
                "best_loss": best_loss if best_loss is not None else None,
                "personal_condition_mode": self._personal_condition_mode(),
                "state_dict": model.modifier_state_dict(include_eeg_head=self.train_eeg),
                "train_eeg": self.train_eeg,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
            }
            torch.save(checkpoint, str(path))

        if save_epoch and epoch is not None:
            save_modifier_checkpoint(ckpt_dir / f"checkpoint_{epoch}.pth")
        if best:
            save_modifier_checkpoint(ckpt_dir / "checkpoint_best.pth")
        if last:
            save_modifier_checkpoint(ckpt_dir / "checkpoint_last.pth")

    def _load_modifier_checkpoint(self, model, optimizer=None, run_key="resume_runid",
                                  names=None, require_eeg_head=False):
        ckpt_dir = self._modifier_dir(run_key)
        names = names or ["checkpoint_best.pth", "checkpoint_last.pth"]
        for name in names:
            checkpoint_path = ckpt_dir / name
            if checkpoint_path.is_file():
                checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
                self._validate_checkpoint_personal_condition_mode(checkpoint, checkpoint_path)
                if require_eeg_head and not self._checkpoint_has_eeg_head(checkpoint):
                    raise RuntimeError(f"ModifierNetwork checkpoint has no trained EEG head: {checkpoint_path}")
                model.load_modifier_state_dict(checkpoint["state_dict"])
                if optimizer is not None and checkpoint.get("optimizer") is not None:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        for state in optimizer.state.values():
                            for key, value in state.items():
                                if torch.is_tensor(value):
                                    state[key] = value.to(self.device)
                    except ValueError:
                        logger.warning(
                            "Skip optimizer state from %s because the modifier parameter set changed.",
                            checkpoint_path,
                        )
                model.to(self.device)
                logger.info("Loaded ModifierNetwork checkpoint: %s", checkpoint_path)
                return checkpoint.get("best_loss", float("inf")), checkpoint.get("epoch", 0)
        raise FileNotFoundError(f"No ModifierNetwork checkpoint found in {ckpt_dir}; tried {names}")

    def _resample_train_batch(self, speaker_audio, speaker_emotion, speaker_3dmm, listener_emotion,
                              listener_eeg=None, listener_eeg_mask=None):
        window_size = self.trainer_cfg.window_size
        clip_length = self.trainer_cfg.clip_length
        s_window_size = self.trainer_cfg.s_ratio * window_size
        has_eeg = listener_eeg is not None and listener_eeg.numel() > 0

        if self.task == "offline":
            motion_lengths = torch.full(
                (speaker_audio.shape[0],),
                min(clip_length, speaker_audio.shape[1]),
                dtype=torch.long,
            )
            eeg_target = listener_eeg[:, motion_lengths[0] - 1] if has_eeg else None
            eeg_mask = listener_eeg_mask[:, motion_lengths[0] - 1] if has_eeg else None
            return speaker_audio, speaker_emotion, speaker_3dmm, listener_emotion, None, motion_lengths, eeg_target, eeg_mask

        if self.task != "online":
            raise ValueError(f"Unknown task type: {self.task}")

        sampled_audio = []
        sampled_emotion = []
        sampled_3dmm = []
        sampled_listener = []
        sampled_past_listener = []
        sampled_eeg = []
        sampled_eeg_mask = []
        for idx in range(speaker_audio.shape[0]):
            seq_length = min(clip_length, speaker_audio.shape[1], listener_emotion.shape[1])
            max_start = max(seq_length - s_window_size, 0)
            cp = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            du = cp + s_window_size

            sampled_audio.append(speaker_audio[idx, cp:du])
            sampled_emotion.append(speaker_emotion[idx, cp:du])
            sampled_3dmm.append(speaker_3dmm[idx, cp:du])
            sampled_past_listener.append(listener_emotion[idx, du - 2 * window_size:du - window_size])
            sampled_listener.append(listener_emotion[idx, du - window_size:du])
            if has_eeg:
                sampled_eeg.append(listener_eeg[idx, du - 1])
                sampled_eeg_mask.append(listener_eeg_mask[idx, du - 1])

        eeg_target = torch.stack(sampled_eeg, dim=0) if has_eeg else None
        eeg_mask = torch.stack(sampled_eeg_mask, dim=0) if has_eeg else None
        return (
            torch.stack(sampled_audio, dim=0),
            torch.stack(sampled_emotion, dim=0),
            torch.stack(sampled_3dmm, dim=0),
            torch.stack(sampled_listener, dim=0),
            torch.stack(sampled_past_listener, dim=0),
            None,
            eeg_target,
            eeg_mask,
        )

    def _split_outputs(self, outputs):
        if isinstance(outputs, dict) and "output_prior" in outputs:
            return outputs["output_prior"], outputs["output_decoder"]
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            return outputs
        raise ValueError("Rewrite-weight training expects diffusion output with prior and decoder branches.")

    def _single_loss(self, model, criterion, speaker_audio, speaker_emotion, speaker_3dmm,
                     listener_emotion, past_listener_emotion, motion_length,
                     personal_3dmm, listener_personality, listener_eeg, listener_eeg_mask, idx):
        input_dict = {
            "speaker_audio_input": speaker_audio[idx:idx + 1],
            "speaker_emotion_input": speaker_emotion[idx:idx + 1],
            "speaker_3dmm_input": speaker_3dmm[idx:idx + 1],
            "listener_emotion_input": listener_emotion[idx:idx + 1],
            "past_listener_emotion": past_listener_emotion[idx:idx + 1] if past_listener_emotion is not None else None,
            "motion_length": motion_length[idx:idx + 1].to(self.device) if motion_length is not None else None,
            "listener_eeg_input": listener_eeg[idx:idx + 1] if listener_eeg is not None else None,
            "listener_eeg_mask": listener_eeg_mask[idx:idx + 1] if listener_eeg_mask is not None else None,
        }
        personal_3dmm_input = personal_3dmm[idx:idx + 1] if personal_3dmm.numel() > 0 else None
        outputs, regular_loss = model(
            x=input_dict,
            p=personal_3dmm_input,
            personality=listener_personality[idx:idx + 1],
        )
        output_prior, output_decoder = self._split_outputs(outputs)
        loss_dict = criterion(output_prior, output_decoder)
        loss = loss_dict["loss_eeg"] if self.train_eeg_head_only else loss_dict["loss"] + regular_loss
        if self.train_eeg_head_only and torch.is_grad_enabled() and not loss.requires_grad:
            raise RuntimeError("loss_eeg has no gradient. Check EEG labels and prediction_eeg.")
        return loss, loss_dict, regular_loss

    def fit(self):
        if self.train_eeg_head_only and self.resumed_training:
            raise ValueError("train_eeg_head_only=True should be launched with resume=false.")

        logger.info("Loading rewrite-weight data module")
        self._ensure_eeg_data_enabled(stage="fit")
        train_loader, val_loader = self.data_module.get_dataloader(stage="fit")
        logger.info("Data module loaded")

        model = self._build_model(stage="fit")
        if self.train_eeg_head_only:
            self._load_modifier_checkpoint_file(
                model,
                self.pretrained_cfg.get("modifier_checkpoint", ""),
                require_eeg_head=False,
            )
        self._load_eeg_head_checkpoint(model, self.pretrained_cfg.get("eeg_head_checkpoint", ""))
        optimizer = self._build_optimizer(model)
        criterion = self._build_criterion()
        writer = SummaryWriter(self.trainer_cfg.tb_dir)

        best_loss = float("inf")
        start_epoch = self.trainer_cfg.start_epoch
        if self.resumed_training:
            best_loss, start_epoch = self._load_modifier_checkpoint(
                model,
                optimizer=optimizer,
                run_key="resume_runid",
                names=["checkpoint_last.pth"],
            )
            logger.info("Resume rewrite-weight training from epoch %s", start_epoch)

        selected_loss_name = "loss_eeg" if self.train_eeg_head_only else "loss"

        for epoch in range(start_epoch, self.trainer_cfg.epochs):
            train_loss, train_prior, train_decoder, train_eeg_loss, train_eeg_valid = self._run_epoch(
                model, train_loader, criterion, optimizer, writer, epoch, train=True
            )
            logger.info(
                "Epoch: %s train_%s: %.5f prior_loss: %.5f decoder_loss: %.5f "
                "eeg_loss: %.5f eeg_valid_ratio: %.5f",
                epoch + 1, selected_loss_name, train_loss, train_prior, train_decoder,
                train_eeg_loss, train_eeg_valid,
            )

            val_loss = train_loss
            val_eeg_loss = train_eeg_loss
            if (epoch + 1) % self.trainer_cfg.val_period == 0:
                val_loss, val_prior, val_decoder, val_eeg_loss, val_eeg_valid = self._run_epoch(
                    model, val_loader, criterion, None, writer, epoch, train=False
                )
                logger.info(
                    "Epoch: %s val_%s: %.5f prior_loss: %.5f decoder_loss: %.5f "
                    "eeg_loss: %.5f eeg_valid_ratio: %.5f",
                    epoch + 1, selected_loss_name, val_loss, val_prior, val_decoder,
                    val_eeg_loss, val_eeg_valid,
                )
                selected_val_loss = val_eeg_loss if self.train_eeg_head_only else val_loss
                if selected_val_loss < best_loss:
                    best_loss = selected_val_loss
                    self._save_checkpoint(
                        model, optimizer, epoch + 1, best=True, save_epoch=True, best_loss=best_loss,
                    )

            if (epoch + 1) % self.trainer_cfg.save_period == 0:
                self._save_checkpoint(model, optimizer, epoch + 1, save_epoch=True, best_loss=best_loss)
            self._save_checkpoint(model, optimizer, epoch + 1, last=True, best_loss=best_loss)

        writer.close()

    def _run_epoch(self, model, data_loader, criterion, optimizer, writer, epoch, train=True):
        whole_losses = AverageMeter()
        prior_losses = AverageMeter()
        decoder_losses = AverageMeter()
        regular_losses = AverageMeter()
        eeg_losses = AverageMeter()
        eeg_valid_ratios = AverageMeter()

        if train and self.train_eeg_head_only:
            model.set_eeg_head_train_mode()
        else:
            model.train(train)
        if model.person_encoder is not None:
            model.person_encoder.eval()
        iterator = tqdm(data_loader)

        for batch_idx, batch in enumerate(iterator):
            if len(batch) == 12:
                (
                    speaker_audio,
                    _,
                    speaker_emotion,
                    speaker_3dmm,
                    _,
                    listener_emotion,
                    _listener_3dmm,
                    personal_3dmm,
                    listener_personality,
                    listener_eeg,
                    listener_eeg_mask,
                    _,
                ) = batch
            elif len(batch) == 10:
                (
                    speaker_audio,
                    _,
                    speaker_emotion,
                    speaker_3dmm,
                    _,
                    listener_emotion,
                    _listener_3dmm,
                    personal_3dmm,
                    listener_personality,
                    _,
                ) = batch
                listener_eeg = listener_eeg_mask = None
            else:
                (
                    speaker_audio,
                    _,
                    speaker_emotion,
                    speaker_3dmm,
                    _,
                    listener_emotion,
                    _listener_3dmm,
                    personal_3dmm,
                    _,
                ) = batch
                listener_personality = speaker_audio.new_zeros((speaker_audio.shape[0], 0))
                listener_eeg = listener_eeg_mask = None
            speaker_audio = speaker_audio.to(self.device)
            speaker_emotion = speaker_emotion.to(self.device)
            speaker_3dmm = speaker_3dmm.to(self.device)
            listener_emotion = listener_emotion.to(self.device)
            personal_3dmm = personal_3dmm.to(self.device)
            listener_personality = listener_personality.to(self.device)
            listener_eeg = listener_eeg.to(self.device) if listener_eeg is not None else None
            listener_eeg_mask = listener_eeg_mask.to(self.device) if listener_eeg_mask is not None else None

            (speaker_audio,
             speaker_emotion,
             speaker_3dmm,
             listener_emotion,
             past_listener_emotion,
             motion_length,
             listener_eeg,
             listener_eeg_mask) = self._resample_train_batch(
                speaker_audio, speaker_emotion, speaker_3dmm, listener_emotion,
                listener_eeg=listener_eeg, listener_eeg_mask=listener_eeg_mask,
            )
            speaker_audio = speaker_audio.to(self.device)
            speaker_emotion = speaker_emotion.to(self.device)
            speaker_3dmm = speaker_3dmm.to(self.device)
            listener_emotion = listener_emotion.to(self.device)
            past_listener_emotion = past_listener_emotion.to(self.device) if past_listener_emotion is not None else None
            motion_length = motion_length.to(self.device) if motion_length is not None else None
            listener_eeg = listener_eeg.to(self.device) if listener_eeg is not None else None
            listener_eeg_mask = listener_eeg_mask.to(self.device) if listener_eeg_mask is not None else None

            batch_size = speaker_audio.shape[0]
            if train:
                optimizer.zero_grad(set_to_none=True)

            batch_loss_value = 0.0
            batch_prior_value = 0.0
            batch_decoder_value = 0.0
            batch_regular_value = 0.0
            batch_eeg_value = 0.0
            batch_eeg_valid_value = 0.0
            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                for idx in range(batch_size):
                    loss, loss_dict, regular_loss = self._single_loss(
                        model,
                        criterion,
                        speaker_audio,
                        speaker_emotion,
                        speaker_3dmm,
                        listener_emotion,
                        past_listener_emotion,
                        motion_length,
                        personal_3dmm,
                        listener_personality,
                        listener_eeg,
                        listener_eeg_mask,
                        idx,
                    )
                    if train:
                        (loss / batch_size).backward()
                    batch_loss_value += loss.detach().item()
                    batch_prior_value += loss_dict["encoded"].detach().item()
                    batch_decoder_value += loss_dict["decoded"].detach().item()
                    batch_regular_value += regular_loss.detach().item()
                    batch_eeg_value += loss_dict["loss_eeg"].detach().item()
                    batch_eeg_valid_value += loss_dict["eeg_valid_ratio"].detach().item()

            if train:
                if self.trainer_cfg.clip_grad:
                    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            batch_loss_value /= batch_size
            batch_prior_value /= batch_size
            batch_decoder_value /= batch_size
            batch_regular_value /= batch_size
            batch_eeg_value /= batch_size
            batch_eeg_valid_value /= batch_size
            whole_losses.update(batch_loss_value, batch_size)
            prior_losses.update(batch_prior_value, batch_size)
            decoder_losses.update(batch_decoder_value, batch_size)
            regular_losses.update(batch_regular_value, batch_size)
            eeg_losses.update(batch_eeg_value, batch_size)
            eeg_valid_ratios.update(batch_eeg_valid_value, batch_size)

            iteration = batch_idx + len(data_loader) * epoch
            if writer is not None:
                prefix = "Train" if train else "Val"
                writer.add_scalar(f"{prefix}/loss", batch_loss_value, iteration)
                writer.add_scalar(f"{prefix}/loss_prior", batch_prior_value, iteration)
                writer.add_scalar(f"{prefix}/loss_decoder", batch_decoder_value, iteration)
                writer.add_scalar(f"{prefix}/regular_loss", batch_regular_value, iteration)
                writer.add_scalar(f"{prefix}/loss_eeg", batch_eeg_value, iteration)
                writer.add_scalar(f"{prefix}/eeg_valid_ratio", batch_eeg_valid_value, iteration)

        return whole_losses.avg, prior_losses.avg, decoder_losses.avg, eeg_losses.avg, eeg_valid_ratios.avg

    def _build_test_windows(self, speaker_audio, speaker_emotion, speaker_3dmm, length):
        clip_len = self.trainer_cfg.clip_length
        window_size = self.trainer_cfg.window_size
        s_window_size = self.trainer_cfg.s_ratio * window_size
        length = int(length.item() if torch.is_tensor(length) else length)

        if self.task == "offline":
            num_windows = max(math.ceil(length / clip_len), 1)
            pad_len = num_windows * clip_len - length
            motion_lengths = torch.tensor(
                [clip_len] * (num_windows - 1) + [length - clip_len * (num_windows - 1)],
                dtype=torch.long,
            )

            def pad_and_rearrange(clip):
                if pad_len > 0:
                    clip = torch.cat((clip, clip.new_zeros((pad_len, clip.shape[-1]))), dim=0)
                return rearrange(clip, "(b l) d -> b l d", b=num_windows)

            return (
                pad_and_rearrange(speaker_audio[:length]),
                pad_and_rearrange(speaker_emotion[:length]),
                pad_and_rearrange(speaker_3dmm[:length]),
                motion_lengths,
            )

        if self.task != "online":
            raise ValueError(f"Unknown task type: {self.task}")

        num_windows = max(math.ceil(length / window_size), 1)

        def pad_online(clip):
            return torch.cat(
                (
                    clip.new_zeros((s_window_size - window_size, clip.shape[-1])),
                    clip[:length],
                    clip.new_zeros((num_windows * window_size - length, clip.shape[-1])),
                ),
                dim=0,
            )

        speaker_audio = pad_online(speaker_audio)
        speaker_emotion = pad_online(speaker_emotion)
        speaker_3dmm = pad_online(speaker_3dmm)

        audio_windows = []
        emotion_windows = []
        coeff_windows = []
        motion_lengths = []
        for idx in range(num_windows):
            start = idx * window_size
            audio_windows.append(speaker_audio[start:start + s_window_size])
            emotion_windows.append(speaker_emotion[start:start + s_window_size])
            coeff_windows.append(speaker_3dmm[start:start + s_window_size])
            motion_lengths.append(window_size if idx < num_windows - 1 else length - idx * window_size)

        return (
            torch.stack(audio_windows, dim=0),
            torch.stack(emotion_windows, dim=0),
            torch.stack(coeff_windows, dim=0),
            torch.tensor(motion_lengths, dtype=torch.long),
        )

    def _apply_personalization(self, model, personal_3dmm, listener_personality):
        personal_3dmm = personal_3dmm.unsqueeze(0).to(self.device) if personal_3dmm.numel() > 0 else None
        listener_personality = listener_personality.unsqueeze(0).to(self.device)
        person_embedding = model.encode_person_condition(
            p=personal_3dmm,
            personality=listener_personality,
        )
        model.kernel = model.hypernet(person_embedding)
        model.apply_weights()

    @staticmethod
    def _eeg_targets_from_motion_lengths(listener_eeg, listener_eeg_mask, motion_lengths):
        if listener_eeg is None or listener_eeg.numel() == 0:
            return None, None
        if listener_eeg_mask is None or listener_eeg_mask.numel() == 0:
            listener_eeg_mask = torch.ones_like(listener_eeg)

        indices = []
        offset = 0
        total_length = listener_eeg.shape[0]
        for motion_length in motion_lengths:
            length = int(motion_length.item() if torch.is_tensor(motion_length) else motion_length)
            last_idx = min(max(offset + max(length, 1) - 1, 0), total_length - 1)
            indices.append(last_idx)
            offset += max(length, 0)
        if not indices:
            return None, None

        index_tensor = torch.tensor(indices, dtype=torch.long)
        return (
            listener_eeg[index_tensor].unsqueeze(0).float(),
            listener_eeg_mask[index_tensor].unsqueeze(0).float(),
        )

    def _predict_windows_once(self, model, speaker_audio, speaker_emotion, speaker_3dmm,
                              motion_lengths, return_eeg=False):
        predictions = []
        eeg_predictions = []
        total_windows = speaker_audio.shape[0]
        for start in range(0, total_windows, self.eval_clip_batch_size):
            end = start + self.eval_clip_batch_size
            input_dict = {
                "speaker_audio_input": speaker_audio[start:end].to(self.device),
                "speaker_emotion_input": speaker_emotion[start:end].to(self.device),
                "speaker_3dmm_input": speaker_3dmm[start:end].to(self.device),
                "motion_length": motion_lengths[start:end].to(self.device),
            }
            outputs = model.main_net(**input_dict)
            predictions.append(outputs["prediction_emotion"].detach().cpu())
            if return_eeg:
                if "prediction_eeg" not in outputs:
                    raise RuntimeError(
                        "trainer.generic.eval_eeg=True but the model did not return prediction_eeg. "
                        "Check configs/<task-section>/model/motion_diffusion.yaml -> eeg_head.enabled."
                    )
                eeg_predictions.append(outputs["prediction_eeg"].detach().cpu())
        if return_eeg:
            return torch.cat(predictions, dim=0), torch.cat(eeg_predictions, dim=0)
        return torch.cat(predictions, dim=0)

    def test(self):
        logger.info("Loading rewrite-weight test data module")
        self._ensure_eeg_data_enabled(stage="test")
        test_loader = self.data_module.get_dataloader(stage="test")
        logger.info("Test data module loaded")

        model = self._build_model(stage="test")
        self._load_modifier_checkpoint(model, run_key="resume_runid", require_eeg_head=self.eval_eeg)
        model.eval()
        if model.person_encoder is not None:
            model.person_encoder.eval()

        logger.info("Loading post processor")
        post_processor = Processor(
            config_name=self.post_config_name,
            clip_len_test=self.post_clip_length,
            device=self.device,
        )
        logger.info("Post processor loaded")

        gt_listener_emotions_all = []
        pred_listener_emotions_all = []
        input_speaker_emotions_all = []
        gt_listener_eeg_all = []
        pred_listener_eeg_all = []
        listener_eeg_mask_all = []

        with torch.inference_mode():
            for batch in tqdm(test_loader):
                if len(batch) == 13:
                    (
                        speaker_audio_clips,
                        _,
                        speaker_emotion_clips,
                        speaker_3dmm_clips,
                        _,
                        listener_emotion_clips,
                        _listener_3dmm_clips,
                        personal_3dmm_clips,
                        listener_personality_clips,
                        listener_eeg_clips,
                        listener_eeg_mask_clips,
                        speaker_seq_lengths,
                        _listener_seq_lengths,
                    ) = batch
                elif len(batch) == 11:
                    (
                        speaker_audio_clips,
                        _,
                        speaker_emotion_clips,
                        speaker_3dmm_clips,
                        _,
                        listener_emotion_clips,
                        _listener_3dmm_clips,
                        personal_3dmm_clips,
                        listener_personality_clips,
                        speaker_seq_lengths,
                        _listener_seq_lengths,
                    ) = batch
                    listener_eeg_clips = listener_eeg_mask_clips = None
                else:
                    (
                        speaker_audio_clips,
                        _,
                        speaker_emotion_clips,
                        speaker_3dmm_clips,
                        _,
                        listener_emotion_clips,
                        _listener_3dmm_clips,
                        personal_3dmm_clips,
                        speaker_seq_lengths,
                        _listener_seq_lengths,
                    ) = batch
                    listener_personality_clips = [
                        speaker_audio.new_zeros((0,))
                        for speaker_audio in speaker_audio_clips
                    ]
                    listener_eeg_clips = listener_eeg_mask_clips = None

                if self.eval_eeg and listener_eeg_clips is None:
                    raise RuntimeError(
                        "trainer.generic.eval_eeg=True but the test dataloader did not return EEG labels."
                    )

                eeg_clips = listener_eeg_clips if self.eval_eeg else [None] * len(speaker_audio_clips)
                eeg_masks = listener_eeg_mask_clips if self.eval_eeg else [None] * len(speaker_audio_clips)

                for (speaker_audio, speaker_emotion, speaker_3dmm, listener_gts,
                     personal_3dmm, listener_personality, listener_eeg, listener_eeg_mask, seq_length) in zip(
                        speaker_audio_clips,
                        speaker_emotion_clips,
                        speaker_3dmm_clips,
                        listener_emotion_clips,
                        personal_3dmm_clips,
                        listener_personality_clips,
                        eeg_clips,
                        eeg_masks,
                        speaker_seq_lengths,
                ):
                    length = int(seq_length.item() if torch.is_tensor(seq_length) else seq_length)
                    input_speaker_emotions_all.append(speaker_emotion[:length])
                    gt_listener_emotions_all.append(listener_gts)

                    windows_audio, windows_emotion, windows_3dmm, motion_lengths = self._build_test_windows(
                        speaker_audio, speaker_emotion, speaker_3dmm, length,
                    )
                    if self.eval_eeg:
                        eeg_target, eeg_mask = self._eeg_targets_from_motion_lengths(
                            listener_eeg, listener_eeg_mask, motion_lengths,
                        )
                        if eeg_target is None:
                            raise RuntimeError("EEG evaluation requested but a sample has no EEG target.")

                    self._apply_personalization(model, personal_3dmm, listener_personality)
                    sample_predictions = []
                    sample_eeg_predictions = []
                    while len(sample_predictions) < self.num_eval_preds:
                        if self.eval_eeg:
                            window_predictions, window_eeg_predictions = self._predict_windows_once(
                                model,
                                windows_audio,
                                windows_emotion,
                                windows_3dmm,
                                motion_lengths,
                                return_eeg=True,
                            )
                        else:
                            window_predictions = self._predict_windows_once(
                                model,
                                windows_audio,
                                windows_emotion,
                                windows_3dmm,
                                motion_lengths,
                            )
                            window_eeg_predictions = None
                        sequence_predictions = rearrange(
                            window_predictions,
                            "b n w d -> n (b w) d",
                        )[:, :length]
                        sample_predictions.extend([prediction for prediction in sequence_predictions])
                        if self.eval_eeg:
                            sequence_eeg_predictions = rearrange(
                                window_eeg_predictions,
                                "b n d -> n b d",
                            )
                            sample_eeg_predictions.extend(
                                [prediction for prediction in sequence_eeg_predictions]
                            )

                    sample_prediction = torch.stack(
                        sample_predictions[:self.num_eval_preds],
                        dim=0,
                    )
                    if self.data_clamp:
                        sample_prediction[:, :, :15] = torch.round(sample_prediction[:, :, :15])
                    pred_listener_emotions_all.append(sample_prediction)

                    if self.eval_eeg:
                        pred_listener_eeg_all.append(torch.stack(
                            sample_eeg_predictions[:self.num_eval_preds],
                            dim=0,
                        ))
                        gt_listener_eeg_all.append(eeg_target)
                        listener_eeg_mask_all.append(eeg_mask)

        if len(pred_listener_emotions_all):
            gt_listener_emotions_all = post_processor.forward(
                prediction_list=pred_listener_emotions_all,
                target_list=gt_listener_emotions_all,
            )

        results_to_save = {
            "GT": gt_listener_emotions_all,
            "PRED": pred_listener_emotions_all,
            "INPUT": input_speaker_emotions_all,
        }
        if self.eval_eeg:
            results_to_save.update({
                "GT_EEG": gt_listener_eeg_all,
                "PRED_EEG": pred_listener_eeg_all,
                "EEG_MASK": listener_eeg_mask_all,
            })
        torch.save(results_to_save, "results.pt")
        logger.info("Saved rewrite-weight results to results.pt")

        results = compute_metrics(
            input_speaker_emotions_all,
            pred_listener_emotions_all,
            gt_listener_emotions_all,
        )
        if self.eval_eeg:
            results.update(compute_eeg_metrics(
                pred_listener_eeg_all,
                gt_listener_eeg_all,
                listener_eeg_mask_all,
            ))
        logger.info(results)
