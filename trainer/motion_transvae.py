import math
from pathlib import Path
from typing import List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_eeg_metrics, compute_metrics
from framework.utils.losses import div_loss
from framework.utils.util import AverageMeter, from_pretrained_checkpoint

os.environ["NUMEXPR_MAX_THREADS"] = '16'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 resumed_training: bool = False,
                 renderer: DictConfig = None,
                 model: DictConfig = None,
                 criterion: DictConfig = None,
                 **kwargs):

        self.renderer_cfg = renderer
        self.model_cfg = model
        self.criterion_cfg = criterion
        self.resumed_training = resumed_training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = kwargs.pop('lr')
        self.optim_cfg = kwargs.pop("optim")
        self.epochs = kwargs.pop("epochs")
        self.gpu_ids = kwargs.pop("gpu_ids")
        self.j = kwargs.pop("j")
        self.max_seq_len = kwargs.pop("max_seq_len")
        self.window_size = kwargs.pop("window_size")
        self.div_p = kwargs.pop("div_p")
        self.task = kwargs.pop("task")
        self.train_eeg_head_only = self._as_bool(kwargs.pop("train_eeg_head_only", False))
        self.eval_eeg = self._as_bool(kwargs.pop("eval_eeg", False))
        self.pretrained_model_checkpoint = kwargs.pop("pretrained_model_checkpoint", "")
        self.num_preds = kwargs.pop("num_preds", 10)
        self.save_results = self._as_bool(kwargs.pop("save_results", True))
        self.eval_facial_metrics = self._as_bool(kwargs.pop("eval_facial_metrics", True))
        self.eval_eeg_metrics = self._as_bool(kwargs.pop("eval_eeg_metrics", True))
        self.metric_threads = int(kwargs.pop("metric_threads", 1))
        self.eval_clip_batch_size = max(int(kwargs.pop("eval_clip_batch_size", 1)), 1)
        self.kwargs = kwargs

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _count_parameters(parameters):
        return sum(parameter.numel() for parameter in parameters)

    @staticmethod
    def _resolve_checkpoint_path(path):
        if path is None or str(path).strip() == "":
            return None
        path = str(path)
        if os.path.isabs(path):
            return path
        return hydra.utils.to_absolute_path(path)

    def get_ckpt_path(self, model, runid="current_runid", epoch=None, best=False, last=False, create_dir=True):
        ckpt_dir = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir")))
        run_id = Path(self.kwargs.get(runid))
        ckpt_dir = str(ckpt_dir / run_id / model.get_model_name())
        if create_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = None
        if epoch is not None:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth")
        if best:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
        if last:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pth")
        assert ckpt_path is not None, "No checkpoint path is provided."
        return ckpt_path

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        return checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    def _load_model_checkpoint(self, model, checkpoint_path, strict=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._checkpoint_state_dict(checkpoint)
        result = model.load_state_dict(state_dict, strict=strict)
        model.to(self.device)
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        if missing:
            logger.warning(f"Missing keys while loading {checkpoint_path}: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys while loading {checkpoint_path}: {unexpected}")
        print(f"Successfully load model checkpoint: {checkpoint_path}")
        if isinstance(checkpoint, dict):
            return checkpoint.get("best_loss", float("inf")), checkpoint.get("epoch", 0)
        return float("inf"), 0

    def _load_pretrained_transvae(self, model):
        checkpoint_path = self._resolve_checkpoint_path(self.pretrained_model_checkpoint)
        if not self.train_eeg_head_only and checkpoint_path is None:
            return
        if self.train_eeg_head_only and checkpoint_path is None:
            raise ValueError(
                "train_eeg_head_only=True requires trainer.pretrained_model_checkpoint. "
                "Point it to a pretrained TransformerVAE checkpoint and launch with resume=false."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing pretrained TransformerVAE checkpoint: {checkpoint_path}")
        self._load_model_checkpoint(model, checkpoint_path, strict=False)
        logger.info(f"Loaded pretrained TransformerVAE checkpoint: {checkpoint_path}")

    def _save_checkpoints(self, model, optimizer, epoch=None, best=False, last=False, best_loss=float("inf")):
        checkpoint = {
            "epoch": epoch,
            "best_loss": best_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, self.get_ckpt_path(model, epoch=epoch, best=best, last=last))

        if getattr(model, "eeg_head", None) is not None:
            eeg_checkpoint = {
                "epoch": epoch,
                "best_loss": best_loss,
                "state_dict": model.eeg_head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(eeg_checkpoint, self.get_ckpt_path(model.eeg_head, epoch=epoch, best=best, last=last))

    def set_data_module(self, data_module):
        self.data_module = data_module

    def data_resample(self, speaker_audio_clips, speaker_video_clips, speaker_emotion_clips, speaker_3dmm_clips,
                      listener_emotion_clips, listener_3dmm_clips, speaker_seq_lengths, listener_seq_lengths,
                      listener_eeg_clips=None, listener_eeg_masks=None):
        speaker_audios = [audio[:L] for audio, L in zip(speaker_audio_clips, speaker_seq_lengths)]
        speaker_videos = [video[:L] for video, L in zip(speaker_video_clips, speaker_seq_lengths)]
        speaker_emotions = [emo[:L] for emo, L in zip(speaker_emotion_clips, speaker_seq_lengths)]
        speaker_3dmm = [param[:L] for param, L in zip(speaker_3dmm_clips, speaker_seq_lengths)]
        listener_emotions = [emo[:L] for emo, L in zip(listener_emotion_clips, listener_seq_lengths)]
        listener_3dmm = [param[:L] for param, L in zip(listener_3dmm_clips, listener_seq_lengths)]
        listener_eegs = listener_eeg_masks_out = None
        if listener_eeg_clips is not None and listener_eeg_masks is not None:
            listener_eegs = [eeg[:L] for eeg, L in zip(listener_eeg_clips, listener_seq_lengths)]
            listener_eeg_masks_out = [mask[:L] for mask, L in zip(listener_eeg_masks, listener_seq_lengths)]
        return (
            speaker_audios,
            speaker_videos,
            speaker_emotions,
            speaker_3dmm,
            listener_emotions,
            listener_3dmm,
            listener_eegs,
            listener_eeg_masks_out,
        )

    def fit(self):
        """
        # relative directory
        root_dir = save/${trainer.task_name}/${data.data_name}/${folder_name}
        # absolute directory
        saving_dir = Path(hydra.utils.to_absolute_path(root_dir))
        # get saving path
        saving_path = str(saving_dir / ...)
        """
        stage = "fit"

        logger.info("Loading data module")
        self.train_loader, self.val_loader = (
            self.data_module.get_dataloader(stage=stage))
        logger.info("Data module loaded")

        logger.info("Loading criterion")
        self.criterion = instantiate(self.criterion_cfg)
        logger.info("Criterion loaded")

        self.main()

    def main(self):
        if self.train_eeg_head_only and self.resumed_training:
            raise ValueError(
                "train_eeg_head_only=True should be launched with resume=false, "
                "so the optimizer and old EEG head checkpoint are not restored."
            )

        model = instantiate(self.model_cfg, _recursive_=False)
        model.to(self.device)
        self._load_pretrained_transvae(model)

        optimizer_params = model.parameters()
        if self.train_eeg_head_only:
            model.freeze_except_eeg_head()
            trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
            trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            frozen_count = self._count_parameters(
                parameter for parameter in model.parameters() if not parameter.requires_grad
            )
            trainable_count = self._count_parameters(trainable_params)
            if len(trainable_params) == 0:
                raise RuntimeError("No trainable parameters found for EEG head-only training.")
            optimizer_params = trainable_params
            logger.info(
                "EEG head-only training enabled. "
                f"Trainable parameters: {trainable_count}; frozen parameters: {frozen_count}"
            )
            logger.info(f"Trainable parameter tensors: {trainable_names}")
            print(
                "EEG head-only training enabled. "
                f"Trainable parameters: {trainable_count}; frozen parameters: {frozen_count}"
            )
            print(f"Trainable parameter tensors: {trainable_names}")

        optimizer = instantiate(self.optim_cfg, lr=self.lr, params=optimizer_params)

        if self.resumed_training:
            checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", last=True)
            from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            lowest_val_loss, start_epoch = self._load_model_checkpoint(model, checkpoint_path, strict=False)
            logger.info(f"Resume training from epoch {start_epoch}")
        else:
            start_epoch = 0
            lowest_val_loss = float('inf')
        print(f"Best validation loss: {lowest_val_loss}")

        for epoch in range(start_epoch, self.epochs):
            train_loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss, div_loss, eeg_loss, eeg_valid_ratio = (
                self.train(model, optimizer)
            )
            logger.info("Epoch: {}  train_loss: {:.5f}  rec_all_loss: {:.5f}  rec_emo_loss: {:.5f}  "
                        "rec_parma_loss: {:.5f}  kld_loss: {:.5f}  div_loss: {:.5f}  "
                        "eeg_loss: {:.5f}  eeg_valid_ratio: {:.5f}"
                  .format(epoch + 1, train_loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss, div_loss,
                          eeg_loss, eeg_valid_ratio))

            if (epoch + 1) % 5 == 0:
                val_loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss, eeg_loss, eeg_valid_ratio = self.val(model)
                logger.info("Epoch: {}  val_loss: {:.5f}  rec_all_loss: {:.5f}  rec_emo_loss: {:.5f}  "
                            "rec_param_loss: {:.5f}  kld_loss: {:.5f}  "
                            "eeg_loss: {:.5f}  eeg_valid_ratio: {:.5f}"
                      .format(epoch + 1, val_loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss,
                              eeg_loss, eeg_valid_ratio))

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    ckpt_path = self.get_ckpt_path(model, best=True)
                    logger.info(f"Saving best checkpoint, val_loss: {lowest_val_loss:.5f}, ckpt_path: {ckpt_path}")
                    self._save_checkpoints(
                        model, optimizer, best=True, epoch=(epoch + 1), best_loss=lowest_val_loss)

                self._save_checkpoints(model, optimizer, epoch=(epoch + 1), best_loss=lowest_val_loss)
                self._save_checkpoints(model, optimizer, last=True, epoch=(epoch + 1), best_loss=lowest_val_loss)

    # Train
    def train(self, model, optimizer):
        losses = AverageMeter()
        rec_losses = AverageMeter()
        rec_emo_losses = AverageMeter()
        rec_param_losses = AverageMeter()
        kld_losses = AverageMeter()
        div_losses = AverageMeter()
        eeg_losses = AverageMeter()
        eeg_valid_ratios = AverageMeter()

        if self.train_eeg_head_only:
            model.set_eeg_head_train_mode()
        else:
            model.train()
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            (
                speaker_audio_clip,
                speaker_video_clip,
                speaker_emotion_clip,
                speaker_3dmm_clip,
                _,
                listener_emotion,
                listener_3dmm,
                speaker_clip_length,
                listener_clip_length,
            ) = batch[:9]
            listener_eeg_clip = listener_eeg_mask = None
            if len(batch) > 9:
                listener_eeg_clip, listener_eeg_mask = batch[9:11]

            if self.model_cfg.task == 'offline':
                (speaker_audio_clip, speaker_video_clip, speaker_emotion_clip, speaker_3dmm_clip,
                 listener_emotion, listener_3dmm, listener_eeg_clip, listener_eeg_mask) = self.data_resample(
                    speaker_audio_clip,
                    speaker_video_clip,
                    speaker_emotion_clip,
                    speaker_3dmm_clip,
                    listener_emotion,
                    listener_3dmm,
                    speaker_clip_length,
                    listener_clip_length,
                    listener_eeg_clips=listener_eeg_clip,
                    listener_eeg_masks=listener_eeg_mask,
                )
            optimizer.zero_grad()
            listener_3dmm_out, listener_emotion_out, distribution, eeg_outputs = model(
                speaker_video_clip,
                speaker_audio_clip,
                motion_lengths=speaker_clip_length,
                speaker_emotion=speaker_emotion_clip,
                speaker_3dmm=speaker_3dmm_clip,
                listener_eeg_input=listener_eeg_clip,
                listener_eeg_mask=listener_eeg_mask,
                return_eeg_outputs=True,
            )

            loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss, eeg_loss, eeg_valid_ratio = self.criterion(
                listener_emotion,
                listener_3dmm,
                listener_emotion_out,
                listener_3dmm_out,
                distribution,
                prediction_eeg=eeg_outputs.get("prediction_eeg"),
                target_eeg=eeg_outputs.get("target_eeg"),
                target_eeg_mask=eeg_outputs.get("target_eeg_mask"),
            )
            if self.train_eeg_head_only:
                loss = eeg_loss
                d_loss = loss.new_tensor(0.0)
                if not loss.requires_grad:
                    raise RuntimeError(
                        "loss_eeg has no gradient. Check that EEG labels are enabled and prediction_eeg is returned."
                    )
            else:
                with torch.no_grad():
                    listener_3dmm_out_, listener_emotion_out_, _ = model(
                        speaker_video_clip,
                        speaker_audio_clip,
                        motion_lengths=speaker_clip_length,
                        return_distribution=False,
                    )
                d_loss = (div_loss(listener_3dmm_out_, listener_3dmm_out) +
                          div_loss(listener_emotion_out_, listener_emotion_out))
                loss = loss + self.div_p * d_loss

            batch_size = len(speaker_video_clip)
            losses.update(loss.data.item(), batch_size)
            rec_losses.update(rec_loss.data.item(), batch_size)
            rec_emo_losses.update(rec_emo_loss.data.item(), batch_size)
            rec_param_losses.update(rec_param_loss.data.item(), batch_size)
            kld_losses.update(kld_loss.data.item(), batch_size)
            div_losses.update(d_loss.data.item(), batch_size)
            eeg_losses.update(eeg_loss.data.item(), batch_size)
            eeg_valid_ratios.update(eeg_valid_ratio.data.item(), batch_size)

            loss.backward()
            optimizer.step()
        return (
            losses.avg,
            rec_losses.avg,
            rec_emo_losses.avg,
            rec_param_losses.avg,
            kld_losses.avg,
            div_losses.avg,
            eeg_losses.avg,
            eeg_valid_ratios.avg,
        )

    # Validation
    def val(self, model):
        losses = AverageMeter()
        rec_losses = AverageMeter()
        rec_emo_losses = AverageMeter()
        rec_param_losses = AverageMeter()
        kld_losses = AverageMeter()
        eeg_losses = AverageMeter()
        eeg_valid_ratios = AverageMeter()
        model.eval()
        model.reset_window_size(8)

        for batch_idx, batch in enumerate(tqdm(self.val_loader)):
            (
                speaker_audio_clip,
                speaker_video_clip,
                speaker_emotion_clip,
                speaker_3dmm_clip,
                _,
                listener_emotion,
                listener_3dmm,
                speaker_clip_length,
                listener_clip_length,
            ) = batch[:9]
            listener_eeg_clip = listener_eeg_mask = None
            if len(batch) > 9:
                listener_eeg_clip, listener_eeg_mask = batch[9:11]
            if self.model_cfg.task == 'offline':
                (speaker_audio_clip, speaker_video_clip, speaker_emotion_clip, speaker_3dmm_clip,
                 listener_emotion, listener_3dmm, listener_eeg_clip, listener_eeg_mask) = self.data_resample(
                    speaker_audio_clip,
                    speaker_video_clip,
                    speaker_emotion_clip,
                    speaker_3dmm_clip,
                    listener_emotion,
                    listener_3dmm,
                    speaker_clip_length,
                    listener_clip_length,
                    listener_eeg_clips=listener_eeg_clip,
                    listener_eeg_masks=listener_eeg_mask,
                )

            with (torch.no_grad()):
                listener_3dmm_out, listener_emotion_out, distribution, eeg_outputs = model(
                    speaker_video_clip,
                    speaker_audio_clip,
                    motion_lengths=speaker_clip_length,
                    speaker_emotion=speaker_emotion_clip,
                    speaker_3dmm=speaker_3dmm_clip,
                    listener_eeg_input=listener_eeg_clip,
                    listener_eeg_mask=listener_eeg_mask,
                    return_eeg_outputs=True,
                )
                loss, rec_loss, rec_emo_loss, rec_param_loss, kld_loss, eeg_loss, eeg_valid_ratio = self.criterion(
                    listener_emotion,
                    listener_3dmm,
                    listener_emotion_out,
                    listener_3dmm_out,
                    distribution,
                    prediction_eeg=eeg_outputs.get("prediction_eeg"),
                    target_eeg=eeg_outputs.get("target_eeg"),
                    target_eeg_mask=eeg_outputs.get("target_eeg_mask"),
                )
                if self.train_eeg_head_only:
                    loss = eeg_loss

                batch_size = len(speaker_video_clip)
                losses.update(loss.data.item(), batch_size)
                rec_losses.update(rec_loss.data.item(), batch_size)
                rec_emo_losses.update(rec_emo_loss.data.item(), batch_size)
                rec_param_losses.update(rec_param_loss.data.item(), batch_size)
                kld_losses.update(kld_loss.data.item(), batch_size)
                eeg_losses.update(eeg_loss.data.item(), batch_size)
                eeg_valid_ratios.update(eeg_valid_ratio.data.item(), batch_size)

        model.reset_window_size(self.window_size)
        return (
            losses.avg,
            rec_losses.avg,
            rec_emo_losses.avg,
            rec_param_losses.avg,
            kld_losses.avg,
            eeg_losses.avg,
            eeg_valid_ratios.avg,
        )

    def pad_to(self, seq: torch.Tensor, length: int) -> torch.Tensor:
        L = seq.shape[0]
        if L < length:
            pad_shape = (length - L, *seq.shape[1:])
            return torch.cat([seq, seq.new_zeros(pad_shape)], dim=0)
        return seq

    def test(self):
        stage = "test"
        data_clamp = self.kwargs.pop("data_clamp")

        model = instantiate(self.model_cfg, _recursive_=False)
        checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", best=True)
        # checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", epoch=30)
        self._load_model_checkpoint(model, checkpoint_path, strict=False)
        model.eval()
        if self.eval_eeg:
            if getattr(model, "eeg_head", None) is None:
                raise RuntimeError(
                    "trainer.eval_eeg=True but configs/<task-section>/model/motion_transvae.yaml has no enabled eeg_head."
                )
            eeg_ckpt_path = self.get_ckpt_path(
                model.eeg_head,
                runid="resume_runid",
                best=True,
                create_dir=False,
            )
            if not os.path.exists(eeg_ckpt_path):
                raise FileNotFoundError(
                    "trainer.eval_eeg=True requires a trained EEGPredictionHead checkpoint. "
                    f"Missing: {eeg_ckpt_path}"
                )
            self._load_model_checkpoint(model.eeg_head, eeg_ckpt_path, strict=True)
            model.eval()

        # Instantiate the renderer only when rendering is requested. Hydra changes the
        # working directory during evaluation, and renderer init loads large external
        # assets that are unnecessary for metrics-only runs.
        renderer = None
        if self.renderer_cfg.do_render:
            renderer = instantiate(
                self.renderer_cfg,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        logger.info("Loading test data module")
        test_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Test data module loaded")

        keep_facial_outputs = self.eval_facial_metrics
        keep_eeg_outputs = self.eval_eeg and (self.save_results or self.eval_eeg_metrics)

        post_processor = None
        post_config_name = self.kwargs.pop("post_config_name")
        post_clip_length = self.kwargs.pop("post_clip_length")
        if keep_facial_outputs:
            logger.info("Loading post processor")
            post_processor = Processor(
                config_name=post_config_name,
                clip_len_test=post_clip_length,
                device=self.device,
            )
            logger.info("Post processor loaded")

        speaker_emotions_input_all = []
        listener_3dmm_preds_lists_all = []
        listener_emotion_preds_lists_all = []
        listener_3dmm_GTs_all = []
        listener_emotion_GTs_all = []
        listener_eeg_preds_lists_all = []
        listener_eeg_GTs_all = []
        listener_eeg_masks_all = []
        max_seq_len = self.max_seq_len
        num_preds = self.num_preds

        for batch_idx, batch in enumerate(tqdm(test_loader)):
            (
                speaker_audio_clips,
                speaker_video_clips,
                speaker_emotion_clips,
                speaker_3dmm_clips,
                listener_video_clips,
                listener_emotions,
                listener_3dmms,
                speaker_clip_lengths,
                listener_clip_lengths,
            ) = batch[:9]
            listener_eeg_clips = listener_eeg_masks = None
            if len(batch) > 9:
                listener_eeg_clips, listener_eeg_masks = batch[9:11]
            if self.eval_eeg and listener_eeg_clips is None:
                raise RuntimeError("trainer.eval_eeg=True but the test dataloader did not return EEG labels.")
            if renderer is not None and len(listener_video_clips) == 0:
                raise RuntimeError(
                    "trainer.renderer.do_render=True but the test dataloader did not return listener video. "
                    "Set data.test_dataset.load_video_l=true when rendering."
                )
            listener_video_iter = listener_video_clips if renderer is not None else [None] * len(speaker_audio_clips)

            listener_3dmm_preds = [] if keep_facial_outputs else None
            listener_emotion_preds = [] if keep_facial_outputs else None
            listener_eeg_preds = [] if keep_eeg_outputs else None
            sample_eeg_targets = [] if keep_eeg_outputs else None
            sample_eeg_masks = [] if keep_eeg_outputs else None
            for i in range(num_preds):
                if i == 0:
                    if keep_facial_outputs:
                        listener_emotion_GTs_all.extend(listener_emotions)
                    if self.eval_facial_metrics:
                        speaker_emotions_input_all.extend(speaker_emotion_clips)

                eeg_clips = listener_eeg_clips if self.eval_eeg else [None] * len(speaker_audio_clips)
                eeg_masks = listener_eeg_masks if self.eval_eeg else [None] * len(speaker_audio_clips)
                for j, (speaker_audio_clip, speaker_video_clip, speaker_emotion_clip, speaker_3dmm_clip,
                        listener_video_clip, speaker_clip_length, listener_eeg_clip, listener_eeg_mask) in (
                        enumerate(zip(speaker_audio_clips, speaker_video_clips, speaker_emotion_clips,
                                      speaker_3dmm_clips, listener_video_iter, speaker_clip_lengths,
                                      eeg_clips, eeg_masks))):

                    speaker_audio_clip_list = []
                    speaker_video_clip_list = []
                    speaker_emotion_clip_list = []
                    speaker_3dmm_clip_list = []
                    listener_eeg_clip_list = []
                    listener_eeg_mask_list = []
                    motion_length_list = []
                    speaker_clip_length_int = int(
                        speaker_clip_length.item() if torch.is_tensor(speaker_clip_length) else speaker_clip_length
                    )

                    # split into sub-clips
                    for k in range(math.ceil(speaker_clip_length_int / max_seq_len)):
                        start_idx = k * max_seq_len
                        end_idx = min((k + 1) * max_seq_len, speaker_clip_length_int)
                        motion_length = end_idx - start_idx
                        motion_length_list.append(motion_length)
                        speaker_audio_clip_list.append(speaker_audio_clip[start_idx:end_idx])
                        speaker_video_clip_list.append(speaker_video_clip[start_idx:end_idx])
                        speaker_emotion_clip_list.append(speaker_emotion_clip[start_idx:end_idx])
                        speaker_3dmm_clip_list.append(speaker_3dmm_clip[start_idx:end_idx])
                        if self.eval_eeg:
                            listener_eeg_clip_list.append(listener_eeg_clip[start_idx:end_idx])
                            listener_eeg_mask_list.append(listener_eeg_mask[start_idx:end_idx])

                    if self.model_cfg.task == 'online':
                        speaker_audio_clip_list[-1] = self.pad_to(speaker_audio_clip_list[-1], max_seq_len)
                        speaker_video_clip_list[-1] = self.pad_to(speaker_video_clip_list[-1], max_seq_len)
                        speaker_emotion_clip_list[-1] = self.pad_to(speaker_emotion_clip_list[-1], max_seq_len)
                        speaker_3dmm_clip_list[-1] = self.pad_to(speaker_3dmm_clip_list[-1], max_seq_len)
                        if self.eval_eeg:
                            listener_eeg_clip_list[-1] = self.pad_to(listener_eeg_clip_list[-1], max_seq_len)
                            listener_eeg_mask_list[-1] = self.pad_to(listener_eeg_mask_list[-1], max_seq_len)
                    speaker_audio_clip_inputs = speaker_audio_clip_list  # List: [tensor([l, d_audio]), ...]
                    speaker_video_clip_inputs = speaker_video_clip_list  # List: [tensor([l, 3, 224, 224]), ...]
                    speaker_emotion_clip_inputs = speaker_emotion_clip_list
                    speaker_3dmm_clip_inputs = speaker_3dmm_clip_list
                    listener_eeg_clip_inputs = listener_eeg_clip_list if self.eval_eeg else None
                    listener_eeg_mask_inputs = listener_eeg_mask_list if self.eval_eeg else None

                    chunk_3dmm_outputs = []
                    chunk_emotion_outputs = []
                    chunk_eeg_outputs = []
                    chunk_eeg_targets = []
                    chunk_eeg_masks = []
                    keep_render_outputs = renderer is not None
                    num_chunks = len(speaker_audio_clip_inputs)
                    for chunk_start in range(0, num_chunks, self.eval_clip_batch_size):
                        chunk_end = min(chunk_start + self.eval_clip_batch_size, num_chunks)
                        with torch.inference_mode():
                            mb_3dmm_out, mb_emotion_out, _, mb_eeg_outputs = model(
                                speaker_video_clip_inputs[chunk_start:chunk_end],
                                speaker_audio_clip_inputs[chunk_start:chunk_end],
                                motion_lengths=torch.tensor(motion_length_list[chunk_start:chunk_end]),
                                speaker_emotion=speaker_emotion_clip_inputs[chunk_start:chunk_end],
                                speaker_3dmm=speaker_3dmm_clip_inputs[chunk_start:chunk_end],
                                listener_eeg_input=listener_eeg_clip_inputs[chunk_start:chunk_end]
                                if self.eval_eeg else None,
                                listener_eeg_mask=listener_eeg_mask_inputs[chunk_start:chunk_end]
                                if self.eval_eeg else None,
                                return_eeg_outputs=True,
                                return_distribution=False,
                            )

                        if keep_facial_outputs or keep_render_outputs:
                            chunk_3dmm_outputs.extend(output.detach().cpu() for output in mb_3dmm_out)
                            chunk_emotion_outputs.extend(output.detach().cpu() for output in mb_emotion_out)

                        if self.eval_eeg:
                            if "prediction_eeg" not in mb_eeg_outputs:
                                raise RuntimeError(
                                    "trainer.eval_eeg=True but the model did not return prediction_eeg."
                                )
                            if "target_eeg" not in mb_eeg_outputs:
                                raise RuntimeError(
                                    "trainer.eval_eeg=True but the model did not return target_eeg."
                                )
                            if keep_eeg_outputs:
                                chunk_eeg_outputs.append(mb_eeg_outputs["prediction_eeg"].detach().cpu())
                                chunk_eeg_targets.append(mb_eeg_outputs["target_eeg"].detach().cpu())
                                chunk_eeg_masks.append(mb_eeg_outputs["target_eeg_mask"].detach().cpu())

                        del mb_3dmm_out, mb_emotion_out, mb_eeg_outputs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    listener_3dmm_out = listener_emotion_out = None
                    if keep_facial_outputs or keep_render_outputs:
                        listener_3dmm_out = torch.cat(chunk_3dmm_outputs, dim=0)[:speaker_clip_length_int]
                        listener_3dmm_out = listener_3dmm_out.unsqueeze(0)
                        listener_emotion_out = torch.cat(chunk_emotion_outputs, dim=0)[:speaker_clip_length_int]
                        listener_emotion_out = listener_emotion_out.unsqueeze(0)
                        if data_clamp:
                            listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])

                    if keep_eeg_outputs:
                        listener_eeg_out = torch.cat(chunk_eeg_outputs, dim=0).unsqueeze(0)
                        listener_eeg_target = torch.cat(chunk_eeg_targets, dim=0).unsqueeze(0)
                        listener_eeg_mask_out = torch.cat(chunk_eeg_masks, dim=0).unsqueeze(0)

                    if renderer is not None and i == 0:  # (batch_idx % 20) == 0
                        listener_video_clip = listener_video_clip[0].to(self.device)
                        val_path = os.path.join('results_videos', 'test')
                        os.makedirs(val_path, exist_ok=True)

                        perm = torch.randperm(listener_video_clip.shape[0])
                        listener_references = listener_video_clip[perm[0]]
                        assert len(listener_references.shape) == 3, \
                            "listener_references.shape should be (3, 224, 224)"
                        renderer.rendering(val_path,
                                           f"batch{str(batch_idx + 1)}",
                                           listener_3dmm_out.to(self.device),
                                           speaker_video_clip,
                                           listener_references,
                                           listener_video_clip)

                    if i == 0:
                        if keep_facial_outputs:
                            listener_3dmm_preds.append(listener_3dmm_out)
                            listener_emotion_preds.append(listener_emotion_out)
                        if keep_eeg_outputs:
                            listener_eeg_preds.append(listener_eeg_out)
                            sample_eeg_targets.append(listener_eeg_target)
                            sample_eeg_masks.append(listener_eeg_mask_out)
                    else:
                        if keep_facial_outputs:
                            listener_3dmm_preds[j] = torch.cat(
                                (listener_3dmm_preds[j], listener_3dmm_out), dim=0)
                            listener_emotion_preds[j] = torch.cat(
                                (listener_emotion_preds[j], listener_emotion_out), dim=0)
                        if keep_eeg_outputs:
                            listener_eeg_preds[j] = torch.cat(
                                (listener_eeg_preds[j], listener_eeg_out), dim=0)

            # listener_3dmm_preds: (num_preds, l, ...)
            if keep_facial_outputs:
                listener_3dmm_preds_lists_all.extend(listener_3dmm_preds)
            # listener_emotion_preds: (num_preds, l, ...)
                listener_emotion_preds_lists_all.extend(listener_emotion_preds)
            if keep_eeg_outputs:
                listener_eeg_preds_lists_all.extend(listener_eeg_preds)
                listener_eeg_GTs_all.extend(sample_eeg_targets)
                listener_eeg_masks_all.extend(sample_eeg_masks)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # listener_emotion_preds_lists_all
        # List: 750 [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]
        # listener_emotion_GTs_all
        # List: 750 [List: [(l', 25), (l'', 25), ...], List: [(l''', 25), (l'''', 25)], ...]
        if keep_facial_outputs and len(listener_emotion_preds_lists_all):
            listener_emotion_GTs_all = post_processor.forward(
                prediction_list=listener_emotion_preds_lists_all,
                target_list=listener_emotion_GTs_all,)
        # listener_emotion_GTs_all
        # List: 750 [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]

        if self.save_results:
            try:
                result_dict = {}
                if keep_facial_outputs:
                    result_dict.update({'GT': listener_emotion_GTs_all, 'PRED': listener_emotion_preds_lists_all})
                if keep_eeg_outputs:
                    result_dict.update({
                        'GT_EEG': listener_eeg_GTs_all,
                        'PRED_EEG': listener_eeg_preds_lists_all,
                        'EEG_MASK': listener_eeg_masks_all,
                    })
                torch.save(result_dict, 'results.pt')
                print("Successfully saved Tensor List")
            except Exception:
                print("Failed to save Tensor List")

        results = {}
        if self.eval_facial_metrics:
            results.update(compute_metrics(
                speaker_emotions_input_all,
                listener_emotion_preds_lists_all,
                listener_emotion_GTs_all,
                threads=self.metric_threads,
            ))
        if self.eval_eeg and self.eval_eeg_metrics:
            results.update(compute_eeg_metrics(
                listener_eeg_preds_lists_all,
                listener_eeg_GTs_all,
                listener_eeg_masks_all,
            ))
        logger.info(results)
