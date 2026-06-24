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
from framework.perreactor_offline import PerReactorOfflineModel, normalize_personal_condition_mode
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
            pretrained: DictConfig = None,
            perreactor: DictConfig = None,
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
        self.pretrained_cfg = pretrained
        self.perreactor_cfg = perreactor or OmegaConf.create({})
        self.batch_size = batch_size
        self.post_config_name = post_config_name
        self.post_clip_length = post_clip_length
        self.data_clamp = data_clamp
        self.num_eval_preds = num_eval_preds
        self.eval_clip_batch_size = eval_clip_batch_size
        self.parallel_eval_preds = parallel_eval_preds
        self.kwargs = kwargs
        self.task = kwargs.get("task", "offline")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_eeg_head_only = self._as_bool(self.trainer_cfg.get("train_eeg_head_only", False))
        self.train_eeg = self._as_bool(self.trainer_cfg.get("train_eeg", False)) or self.train_eeg_head_only
        self.eval_eeg = self._as_bool(self.trainer_cfg.get("eval_eeg", False))
        self.pretrained_eeg_head_loaded = False

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
            latent_embedder=model_cfg.latent_embedder if hasattr(model_cfg, "latent_embedder") else None,
            audio_encoder=model_cfg.audio_encoder if hasattr(model_cfg, "audio_encoder") else None,
            **self.kwargs,
            _recursive_=False,
        )
        model.to(self.device)
        self._load_pretrained_diffusion(model)
        return model

    def _load_pretrained_diffusion(self, model):
        if self.pretrained_cfg is None:
            raise ValueError("Missing trainer.pretrained configuration for S-PerReactor.")

        def load_required(path, module, label):
            checkpoint_path = hydra.utils.to_absolute_path(path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Missing pretrained {label} checkpoint: {checkpoint_path}")
            from_pretrained_checkpoint(checkpoint_path, module, self.device)

        if getattr(model, "diffusion_prior", None) is not None:
            load_required(self.pretrained_cfg.diffusion_prior, model.diffusion_prior.model, "DiffusionPriorNetwork")
        load_required(self.pretrained_cfg.diffusion_decoder, model.diffusion_decoder.model, "TransformerDenoiser")

    def _load_eeg_head_checkpoint(self, model, path):
        checkpoint_path = self._resolve_checkpoint_path(path)
        if checkpoint_path is None:
            return False
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
        if getattr(model, "eeg_head", None) is None:
            raise RuntimeError("Cannot load EEG head checkpoint because diffusion.eeg_head is disabled.")
        model.eeg_head.load_state_dict(state_dict)
        model.to(self.device)
        logger.info("Loaded EEG head checkpoint: %s", checkpoint_path)
        return True

    def _build_model(self, stage):
        diffusion = self._build_diffusion(stage)
        eeg_head_checkpoint = self.pretrained_cfg.get("eeg_head_checkpoint", "")
        should_load_eeg_head = bool(str(eeg_head_checkpoint).strip()) and (
            self.train_eeg or (stage == "test" and self.eval_eeg)
        )
        self.pretrained_eeg_head_loaded = (
            self._load_eeg_head_checkpoint(diffusion, eeg_head_checkpoint)
            if should_load_eeg_head else False
        )
        model = PerReactorOfflineModel(
            diffusion=diffusion,
            personal_feature_dim=self.perreactor_cfg.get("personal_feature_dim", 25),
            personal_hidden_dim=self.perreactor_cfg.get("personal_hidden_dim", 128),
            personal_embed_dim=self.perreactor_cfg.get("personal_embed_dim", 512),
            k=self.perreactor_cfg.get("k", 2),
            dropout=self.perreactor_cfg.get("dropout", 0.1),
            personal_condition_mode=self.perreactor_cfg.get("personal_condition_mode", "history_only"),
            personality_input_dim=self.perreactor_cfg.get("personality_input_dim", 5),
            personality_hidden_dim=self.perreactor_cfg.get("personality_hidden_dim", 128),
            personality_dropout=self.perreactor_cfg.get("personality_dropout", 0.1),
            personality_fusion_hidden_dim=self.perreactor_cfg.get("personality_fusion_hidden_dim", 512),
            freeze_backbone=self._as_bool(self.perreactor_cfg.get("freeze_backbone", True)),
        )
        model.to(self.device)
        return model

    def _build_criterion(self):
        return partial(getattr(rewrite_losses, self.criterion_cfg.type), **self.criterion_cfg.args)

    def _build_optimizer(self, model):
        cfg = self.perreactor_cfg.get("optimizer", {})
        if self.train_eeg_head_only:
            if not model.has_eeg_head():
                raise ValueError("trainer.generic.train_eeg_head_only=True requires a diffusion EEG head.")
            model.freeze_except_eeg_head()
            parameters = list(model.eeg_head().parameters())
        else:
            if self.train_eeg:
                model.set_eeg_head_requires_grad(True)
            parameters = model.adapter_parameters(include_eeg_head=self.train_eeg)
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        if len(parameters) == 0:
            raise ValueError("No trainable parameters found for S-PerReactor optimizer.")
        return optim.AdamW(
            parameters,
            lr=cfg.get("lr", 0.0001),
            weight_decay=cfg.get("weight_decay", 5e-4),
            eps=cfg.get("eps", 1e-8),
        )

    def _adapter_dir(self, run_key="current_runid"):
        ckpt_root = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir")))
        run_id = self.kwargs.get(run_key) or self.kwargs.get("current_runid")
        ckpt_dir = ckpt_root / str(run_id) / "PerReactor"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def _save_checkpoint(self, model, optimizer, epoch=None, best=False, last=False,
                         save_epoch=False, best_loss=float("inf")):
        ckpt_dir = self._adapter_dir("current_runid")

        def save_adapter_checkpoint(path):
            checkpoint = {
                "epoch": epoch if epoch is not None else None,
                "best_loss": best_loss if best_loss is not None else None,
                "personal_condition_mode": model.personal_condition_mode,
                "train_eeg": self.train_eeg,
                "train_eeg_head_only": self.train_eeg_head_only,
                "state_dict": model.adapter_state_dict(include_eeg_head=self.train_eeg),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
            }
            torch.save(checkpoint, str(path))

        if save_epoch and epoch is not None:
            save_adapter_checkpoint(ckpt_dir / f"checkpoint_{epoch}.pth")
        if best:
            save_adapter_checkpoint(ckpt_dir / "checkpoint_best.pth")
        if last:
            save_adapter_checkpoint(ckpt_dir / "checkpoint_last.pth")

    @staticmethod
    def _checkpoint_has_eeg_head(checkpoint):
        state_dict = checkpoint.get("state_dict", checkpoint)
        return any(key.startswith("eeg_head.") for key in state_dict)

    def _validate_checkpoint_personal_condition_mode(self, model, checkpoint, checkpoint_path):
        checkpoint_mode = checkpoint.get("personal_condition_mode")
        if checkpoint_mode is None:
            logger.warning(
                "S-PerReactor checkpoint %s does not record personal_condition_mode; assuming compatible.",
                checkpoint_path,
            )
            return
        checkpoint_mode = normalize_personal_condition_mode(checkpoint_mode)
        if checkpoint_mode != model.personal_condition_mode:
            raise ValueError(
                "S-PerReactor checkpoint personal_condition_mode mismatch: "
                f"checkpoint={checkpoint_mode}, current={model.personal_condition_mode}. "
                "Use the same trainer.perreactor.personal_condition_mode as training."
            )

    def _load_adapter_checkpoint_path(
            self,
            model,
            checkpoint_path,
            optimizer=None,
            strict=True,
            require_eeg_head=False,
    ):
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Missing S-PerReactor adapter checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._validate_checkpoint_personal_condition_mode(model, checkpoint, checkpoint_path)
        if require_eeg_head and not self._checkpoint_has_eeg_head(checkpoint):
            raise RuntimeError(
                "EEG evaluation requested but the S-PerReactor checkpoint does not contain eeg_head.* weights. "
                "Provide trainer.pretrained.eeg_head_checkpoint or evaluate a checkpoint trained with train_eeg=True."
            )
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_adapter_state_dict(state_dict, strict=strict)
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        model.to(self.device)
        logger.info("Loaded S-PerReactor checkpoint: %s", checkpoint_path)
        return checkpoint.get("best_loss", float("inf")), checkpoint.get("epoch", 0)

    def _load_adapter_checkpoint(
            self,
            model,
            optimizer=None,
            run_key="resume_runid",
            names=None,
            strict=True,
            require_eeg_head=False,
    ):
        ckpt_dir = self._adapter_dir(run_key)
        names = names or ["checkpoint_best.pth", "checkpoint_last.pth"]
        for name in names:
            checkpoint_path = ckpt_dir / name
            if checkpoint_path.is_file():
                return self._load_adapter_checkpoint_path(
                    model,
                    str(checkpoint_path),
                    optimizer=optimizer,
                    strict=strict,
                    require_eeg_head=require_eeg_head,
                )
        raise FileNotFoundError(f"No S-PerReactor checkpoint found in {ckpt_dir}; tried {names}")

    def _resample_train_batch(self, speaker_audio, speaker_emotion, speaker_3dmm, listener_emotion,
                              listener_eeg=None, listener_eeg_mask=None):
        if self.task != "offline":
            raise ValueError(f"S-PerReactor offline trainer only supports task=offline, got {self.task}")
        clip_length = self.trainer_cfg.clip_length
        motion_lengths = torch.full(
            (speaker_audio.shape[0],),
            min(clip_length, speaker_audio.shape[1]),
            dtype=torch.long,
        )
        has_eeg = listener_eeg is not None and listener_eeg.numel() > 0
        eeg_target = listener_eeg[:, motion_lengths[0] - 1] if has_eeg else None
        eeg_mask = listener_eeg_mask[:, motion_lengths[0] - 1] if has_eeg else None
        return speaker_audio, speaker_emotion, speaker_3dmm, listener_emotion, None, motion_lengths, eeg_target, eeg_mask

    @staticmethod
    def _split_outputs(outputs):
        if isinstance(outputs, dict) and "output_prior" in outputs:
            return outputs["output_prior"], outputs["output_decoder"]
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            return outputs
        raise ValueError("S-PerReactor training expects diffusion output with prior and decoder branches.")

    def fit(self):
        logger.info("Loading S-PerReactor data module")
        self._ensure_eeg_data_enabled(stage="fit")
        train_loader, val_loader = self.data_module.get_dataloader(stage="fit")
        logger.info("Data module loaded")

        model = self._build_model(stage="fit")
        criterion = self._build_criterion()

        best_loss = float("inf")
        start_epoch = self.trainer_cfg.start_epoch
        if self.train_eeg_head_only:
            if self.resumed_training:
                raise ValueError(
                    "trainer.generic.train_eeg_head_only=True starts a second-stage EEG run from "
                    "trainer.pretrained.adapter_checkpoint; resume that EEG run with train_eeg_head_only=False."
                )
            adapter_checkpoint = self.pretrained_cfg.get("adapter_checkpoint", "")
            if adapter_checkpoint is None or str(adapter_checkpoint).strip() == "":
                raise ValueError(
                    "trainer.generic.train_eeg_head_only=True requires trainer.pretrained.adapter_checkpoint "
                    "to initialize the trained S-PerReactor adapter."
                )
            self._load_adapter_checkpoint_path(model, adapter_checkpoint, strict=True, require_eeg_head=False)

        optimizer = self._build_optimizer(model)
        if self.resumed_training:
            best_loss, loaded_epoch = self._load_adapter_checkpoint(
                model,
                optimizer=optimizer,
                run_key="resume_runid",
                names=["checkpoint_last.pth", "checkpoint_best.pth"],
            )
            start_epoch = loaded_epoch or start_epoch

        writer = SummaryWriter(log_dir=str(Path(self.trainer_cfg.tb_dir)))
        for epoch in range(start_epoch, self.trainer_cfg.epochs):
            train_loss, train_prior, train_decoder, train_eeg_loss, train_eeg_valid = self._run_epoch(
                model, train_loader, criterion, optimizer, writer, epoch, train=True
            )
            logger.info(
                "Epoch: %s train_loss: %.5f prior_loss: %.5f decoder_loss: %.5f "
                "eeg_loss: %.5f eeg_valid_ratio: %.5f",
                epoch + 1, train_loss, train_prior, train_decoder, train_eeg_loss, train_eeg_valid,
            )

            if (epoch + 1) % self.trainer_cfg.val_period == 0:
                val_loss, val_prior, val_decoder, val_eeg_loss, val_eeg_valid = self._run_epoch(
                    model, val_loader, criterion, None, writer, epoch, train=False
                )
                logger.info(
                    "Epoch: %s val_loss: %.5f prior_loss: %.5f decoder_loss: %.5f "
                    "eeg_loss: %.5f eeg_valid_ratio: %.5f",
                    epoch + 1, val_loss, val_prior, val_decoder, val_eeg_loss, val_eeg_valid,
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    self._save_checkpoint(model, optimizer, epoch + 1, best=True, save_epoch=True, best_loss=best_loss)

            if (epoch + 1) % self.trainer_cfg.save_period == 0:
                self._save_checkpoint(model, optimizer, epoch + 1, save_epoch=True, best_loss=best_loss)
            self._save_checkpoint(model, optimizer, epoch + 1, last=True, best_loss=best_loss)
        writer.close()

    def _run_epoch(self, model, data_loader, criterion, optimizer, writer, epoch, train=True):
        whole_losses = AverageMeter()
        prior_losses = AverageMeter()
        decoder_losses = AverageMeter()
        eeg_losses = AverageMeter()
        eeg_valid_ratios = AverageMeter()

        if train:
            if self.train_eeg_head_only:
                model.freeze_except_eeg_head()
                model.set_eeg_head_train_mode()
            else:
                model.train(True)
                if self.train_eeg:
                    model.set_eeg_head_train_mode()
        else:
            model.eval()
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
                    listener_personal_clip,
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
                    listener_personal_clip,
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
                    listener_personal_clip,
                    _,
                ) = batch
                listener_personality = None
                listener_eeg = listener_eeg_mask = None

            speaker_audio = speaker_audio.to(self.device)
            speaker_emotion = speaker_emotion.to(self.device)
            speaker_3dmm = speaker_3dmm.to(self.device)
            listener_emotion = listener_emotion.to(self.device)
            listener_personal_clip = listener_personal_clip.to(self.device)
            listener_personality = listener_personality.to(self.device) if listener_personality is not None else None
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
            motion_length = motion_length.to(self.device)
            if train:
                optimizer.zero_grad(set_to_none=True)

            input_dict = {
                "speaker_audio_input": speaker_audio,
                "speaker_emotion_input": speaker_emotion,
                "speaker_3dmm_input": speaker_3dmm,
                "listener_emotion_input": listener_emotion,
                "past_listener_emotion": past_listener_emotion,
                "motion_length": motion_length,
                "listener_eeg_input": listener_eeg,
                "listener_eeg_mask": listener_eeg_mask,
            }
            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                outputs = model(
                    x=input_dict,
                    listener_personal_clip=listener_personal_clip,
                    personality=listener_personality,
                )
                output_prior, output_decoder = self._split_outputs(outputs)
                loss_dict = criterion(output_prior, output_decoder)
                loss = loss_dict["loss_eeg"] if self.train_eeg_head_only else loss_dict["loss"]
                if train and self.train_eeg_head_only and not loss.requires_grad:
                    raise RuntimeError(
                        "train_eeg_head_only=True selected loss_eeg, but it has no gradient. "
                        "Check that EEG labels are loaded and the diffusion model has an EEG head."
                    )
                if train:
                    loss.backward()

            if train:
                if self.trainer_cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        1.0,
                    )
                optimizer.step()

            batch_size = speaker_audio.shape[0]
            whole_losses.update(loss.detach().item(), batch_size)
            prior_losses.update(loss_dict["encoded"].detach().item(), batch_size)
            decoder_losses.update(loss_dict["decoded"].detach().item(), batch_size)
            eeg_losses.update(loss_dict["loss_eeg"].detach().item(), batch_size)
            eeg_valid_ratios.update(loss_dict["eeg_valid_ratio"].detach().item(), batch_size)

            iteration = batch_idx + len(data_loader) * epoch
            if writer is not None:
                prefix = "Train" if train else "Val"
                writer.add_scalar(f"{prefix}/loss", loss.detach().item(), iteration)
                writer.add_scalar(f"{prefix}/loss_prior", loss_dict["encoded"].detach().item(), iteration)
                writer.add_scalar(f"{prefix}/loss_decoder", loss_dict["decoded"].detach().item(), iteration)
                writer.add_scalar(f"{prefix}/loss_eeg", loss_dict["loss_eeg"].detach().item(), iteration)
                writer.add_scalar(f"{prefix}/eeg_valid_ratio", loss_dict["eeg_valid_ratio"].detach().item(), iteration)

        return whole_losses.avg, prior_losses.avg, decoder_losses.avg, eeg_losses.avg, eeg_valid_ratios.avg

    def _build_test_windows(self, speaker_audio, speaker_emotion, speaker_3dmm, length):
        clip_len = self.trainer_cfg.clip_length
        length = int(length.item() if torch.is_tensor(length) else length)
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
                              motion_lengths, listener_personal_embed, return_eeg=False):
        predictions = []
        eeg_predictions = []
        total_windows = speaker_audio.shape[0]
        for start in range(0, total_windows, self.eval_clip_batch_size):
            end = min(start + self.eval_clip_batch_size, total_windows)
            batch_size = end - start
            input_dict = {
                "speaker_audio_input": speaker_audio[start:end].to(self.device),
                "speaker_emotion_input": speaker_emotion[start:end].to(self.device),
                "speaker_3dmm_input": speaker_3dmm[start:end].to(self.device),
                "motion_length": motion_lengths[start:end].to(self.device),
            }
            personal_embed = listener_personal_embed.expand(batch_size, -1)
            outputs = model(x=input_dict, listener_personal_embed=personal_embed)
            predictions.append(outputs["prediction_emotion"].detach().cpu())
            if return_eeg:
                if "prediction_eeg" not in outputs:
                    raise RuntimeError("EEG evaluation requested but model did not return prediction_eeg.")
                eeg_predictions.append(outputs["prediction_eeg"].detach().cpu())
        if return_eeg:
            return torch.cat(predictions, dim=0), torch.cat(eeg_predictions, dim=0)
        return torch.cat(predictions, dim=0)

    def test(self):
        logger.info("Loading S-PerReactor test data module")
        self._ensure_eeg_data_enabled(stage="test")
        test_loader = self.data_module.get_dataloader(stage="test")
        logger.info("Test data module loaded")

        model = self._build_model(stage="test")
        self._load_adapter_checkpoint(
            model,
            run_key="resume_runid",
            require_eeg_head=self.eval_eeg and not self.pretrained_eeg_head_loaded,
        )
        if self.eval_eeg and not model.has_eeg_head():
            raise RuntimeError("trainer.generic.eval_eeg=True requires an EEG head in the diffusion model.")
        model.eval()

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
                        listener_personal_clips,
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
                        listener_personal_clips,
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
                        listener_personal_clips,
                        speaker_seq_lengths,
                        _listener_seq_lengths,
                    ) = batch
                    listener_eeg_clips = listener_eeg_mask_clips = None
                    listener_personality_clips = [None] * len(speaker_audio_clips)

                if self.eval_eeg and listener_eeg_clips is None:
                    raise RuntimeError("trainer.generic.eval_eeg=True but the test dataloader did not return EEG labels.")
                eeg_clips = listener_eeg_clips if self.eval_eeg else [None] * len(speaker_audio_clips)
                eeg_masks = listener_eeg_mask_clips if self.eval_eeg else [None] * len(speaker_audio_clips)

                for (speaker_audio, speaker_emotion, speaker_3dmm, listener_gts,
                     listener_personal_clip, listener_personality, listener_eeg, listener_eeg_mask, seq_length) in zip(
                        speaker_audio_clips,
                        speaker_emotion_clips,
                        speaker_3dmm_clips,
                        listener_emotion_clips,
                        listener_personal_clips,
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

                    listener_personal_clip = (
                        listener_personal_clip.unsqueeze(0).to(self.device)
                        if listener_personal_clip is not None and listener_personal_clip.numel() > 0
                        else None
                    )
                    listener_personality = (
                        listener_personality.unsqueeze(0).to(self.device)
                        if listener_personality is not None and listener_personality.numel() > 0
                        else None
                    )
                    listener_personal_embed = model.encode_person_condition(
                        listener_personal_clip=listener_personal_clip,
                        personality=listener_personality,
                    )
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
                                listener_personal_embed,
                                return_eeg=True,
                            )
                        else:
                            window_predictions = self._predict_windows_once(
                                model,
                                windows_audio,
                                windows_emotion,
                                windows_3dmm,
                                motion_lengths,
                                listener_personal_embed,
                            )
                            window_eeg_predictions = None
                        sequence_predictions = rearrange(window_predictions, "b n w d -> n (b w) d")[:, :length]
                        sample_predictions.extend([prediction for prediction in sequence_predictions])
                        if self.eval_eeg:
                            sequence_eeg_predictions = rearrange(window_eeg_predictions, "b n d -> n b d")
                            sample_eeg_predictions.extend([prediction for prediction in sequence_eeg_predictions])

                    sample_prediction = torch.stack(sample_predictions[:self.num_eval_preds], dim=0)
                    if self.data_clamp:
                        sample_prediction[:, :, :15] = torch.round(sample_prediction[:, :, :15])
                    pred_listener_emotions_all.append(sample_prediction)

                    if self.eval_eeg:
                        pred_listener_eeg_all.append(torch.stack(sample_eeg_predictions[:self.num_eval_preds], dim=0))
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
        logger.info("Saved S-PerReactor results to results.pt")

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
