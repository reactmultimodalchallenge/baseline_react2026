"""
Code adapted from:
https://github.com/BarqueroGerman/BeLFusion
"""
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import os
from einops import rearrange
from omegaconf import DictConfig
from hydra.utils import instantiate
from framework.motion_diffusion.diffusion.diffusion_decoder.transformer_denoiser import TransformerDenoiser, \
    lengths_to_mask
from framework.motion_diffusion.diffusion.diffusion_prior.transformer_prior import DiffusionPriorNetwork
from framework.motion_diffusion.diffusion.gaussian_diffusion import PriorLatentDiffusion, DecoderLatentDiffusion
from framework.motion_diffusion.diffusion.resample import UniformSampler
from framework.motion_diffusion.diffusion.rnn import LatentEmbedder
from framework.utils.util import from_pretrained_checkpoint, save_checkpoint


class EEGPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=14, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_model_name(self):
        return self.__class__.__name__


class BaseLatentModel(nn.Module):
    def __init__(self, cfg, emb_preprocessing=False, freeze_encoder=True, **kwargs):
        super(BaseLatentModel, self).__init__()
        self.emb_preprocessing = emb_preprocessing
        self.freeze_encoder = freeze_encoder
        def_dtype = torch.get_default_dtype()

        self.audio_encoder = instantiate(cfg.audio_encoder)
        if cfg.latent_embedder is not None:
            self.latent_embedder = instantiate(cfg.latent_embedder)
            model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.latent_embedder.checkpoint_path)
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            self.latent_embedder.load_state_dict(state_dict)
            print(f"Successfully loaded latent embedder from {model_path}")
        else:
            self.latent_embedder = LatentEmbedder()

        if self.freeze_encoder:  # freeze modules
            for para in self.latent_embedder.parameters():
                para.requires_grad = False

        torch.set_default_dtype(def_dtype)
        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, "Cannot deepcopy LatentUNetMatcher if init_params is None."
        # I can't deep copy this class. I need to do this trick to make the deepcopy of everything
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing:
            return 2 * (emb - stats["min"]) / (stats["max"] - stats["min"]) - 1
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")

    # override checkpointing
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    # override eval and train
    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class PriorLatentMatcher(BaseLatentModel):
    def __init__(self,
                 conf: DictConfig = None,
                 module_dict_cfg: DictConfig = None,
                 stage: str = 'fit',
                 task: str = 'online',
                 **kwargs):
        cfg = conf.args
        super(PriorLatentMatcher, self).__init__(
            module_dict_cfg,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
            **kwargs,
        )

        self.stage = stage
        self.task = task
        self.token_len = cfg.token_len
        self.window_size = cfg.get("window_size", 30)
        self.s_ratio = cfg.get("s_ratio", 2)
        self.s_window_size = cfg.get("s_window_size", self.window_size * self.s_ratio)

        self.init_params = {
            "audio_dim": cfg.get("audio_dim", 768),
            "window_size": self.s_window_size,
            "_3dmm_dim": cfg.get("_3dmm_dim", 58),
            "speaker_emb_dim": cfg.get("speaker_emb_dim", 512),
            "latent_dim": cfg.get("latent_dim", 512),
            "depth": cfg.get("depth", 4),
            "num_time_layers": cfg.get("num_time_layers", 2),
            "num_time_embeds": cfg.get("num_time_embeds", 1),
            "num_time_emb_channels": cfg.get("num_time_emb_channels", 64),
            "time_last_act": cfg.get("time_last_act", False),
            "use_learned_query": cfg.get("use_learned_query", True),
            "s_audio_cond_drop_prob": cfg.get("s_audio_cond_drop_prob", 0.2),
            "s_latentemb_cond_drop_prob": cfg.get("s_latentemb_cond_drop_prob", 1.0),
            "s_3dmm_cond_drop_prob": cfg.get("s_3dmm_cond_drop_prob", 0.2),
            "guidance_scale": cfg.get("guidance_scale", 1.0),
            "dim_head": cfg.get("dim_head", 64),
            "heads": cfg.get("heads", 8),
            "ff_mult": cfg.get("ff_mult", 4),
            "norm_in": cfg.get("norm_in", False),
            "norm_out": cfg.get("norm_out", True),
            "attn_dropout": cfg.get("attn_dropout", 0.0),
            "ff_dropout": cfg.get("ff_dropout", 0.0),
            "final_proj": cfg.get("final_proj", True),
            "normformer": cfg.get("normformer", False),
            "rotary_emb": cfg.get("rotary_emb", True),
        }
        self.model = DiffusionPriorNetwork(**self.init_params)

        self.prior_diffusion = PriorLatentDiffusion(
            conf.scheduler,
            conf.scheduler.num_train_timesteps,
            conf.scheduler.num_inference_timesteps,
        )
        self.schedule_sampler = UniformSampler(self.prior_diffusion)
        self.num_preds = conf.scheduler.num_preds

    def _select_training_windows(self,
                                 speaker_audio_input,
                                 speaker_emotion_input,
                                 speaker_3dmm_input,
                                 listener_emotion_input):
        speaker_len = speaker_audio_input.shape[1]
        listener_len = listener_emotion_input.shape[1]

        if speaker_len > self.s_window_size:
            max_start = speaker_len - self.s_window_size
            window_start = torch.randint(0, max_start + 1, (1,), device=speaker_audio_input.device).item()
        else:
            window_start = 0

        speaker_audio_input = speaker_audio_input[:, window_start:window_start + self.s_window_size]
        speaker_emotion_input = speaker_emotion_input[:, window_start:window_start + self.s_window_size]
        speaker_3dmm_input = speaker_3dmm_input[:, window_start:window_start + self.s_window_size]

        listener_start = min(window_start, max(listener_len - self.window_size, 0))
        listener_emotion_input = listener_emotion_input[:, listener_start:listener_start + self.window_size]

        return speaker_audio_input, speaker_emotion_input, speaker_3dmm_input, listener_emotion_input

    def _forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            **kwargs,
    ):
        if self.stage == "test":
            raise RuntimeError("PriorLatentMatcher is trained and checkpointed only; inference still uses decoder output.")

        (speaker_audio_input,
         speaker_emotion_input,
         speaker_3dmm_input,
         listener_emotion_input) = self._select_training_windows(
            speaker_audio_input,
            speaker_emotion_input,
            speaker_3dmm_input,
            listener_emotion_input,
        )

        with torch.no_grad():
            s_audio_encodings = self.audio_encoder._encode(speaker_audio_input)
            s_audio_encodings = s_audio_encodings.repeat_interleave(self.num_preds, dim=0)

            s_latent_embed = self.latent_embedder.encode(speaker_emotion_input).unsqueeze(1)
            s_latent_embed = s_latent_embed.repeat_interleave(self.num_preds, dim=0)

            s_3dmm_encodings = speaker_3dmm_input.repeat_interleave(self.num_preds, dim=0)

            listener_latent_embed = self.latent_embedder.encode(listener_emotion_input).unsqueeze(1)
            listener_latent_embed = listener_latent_embed.repeat_interleave(self.num_preds, dim=0)

            model_kwargs = {
                "speaker_audio_encodings": s_audio_encodings,
                "speaker_latent_emb": s_latent_embed,
                "speaker_3dmm_encodings": s_3dmm_encodings,
            }

        t, _ = self.schedule_sampler.sample(listener_latent_embed.shape[0] // self.num_preds,
                                            listener_latent_embed.device)
        output_prior = self.prior_diffusion.denoise(
            self.model,
            listener_latent_embed,
            t,
            model_kwargs=model_kwargs,
        )
        return output_prior

    def forward(self, **kwargs):
        return self._forward(**kwargs)


class DecoderLatentMatcher(BaseLatentModel):
    def __init__(self,
                 conf: DictConfig = None,
                 module_dict_cfg: DictConfig = None,
                 stage: str = 'fit',
                 task: str = 'online',
                 **kwargs):
        cfg = conf.args
        super(DecoderLatentMatcher, self).__init__(
            module_dict_cfg,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
            **kwargs,
        )

        self.stage = stage
        self.task = task
        self.token_len = cfg.token_len
        self.window_size = cfg.get("window_size", 30)
        self.s_ratio = cfg.get("s_ratio", 2)
        self.emotion_dim = cfg.get("nfeats", 25)
        self.encode_emotion = cfg.get("encode_emotion", False)
        self.encode_3dmm = cfg.get("encode_3dmm", False)

        self.init_params = {
            "task": task,
            "window_size": self.window_size,
            "encode_emotion": self.encode_emotion,
            "encode_3dmm": self.encode_3dmm,
            "ablation_skip_connection": cfg.get("ablation_skip_connection", True),
            "nfeats": cfg.get("nfeats", 25),
            "latent_dim": cfg.get("latent_dim", 512),
            "ff_size": cfg.get("ff_size", 1024),
            "num_layers": cfg.get("num_layers", 6),
            "num_heads": cfg.get("num_heads", 4),
            "dropout": cfg.get("dropout", 0.1),
            "normalize_before": cfg.get("normalize_before", False),
            "activation": cfg.get("activation", "gelu"),
            "flip_sin_to_cos": cfg.get("flip_sin_to_cos", True),
            "return_intermediate_dec": cfg.get("return_intermediate_dec", False),
            "position_embedding": cfg.get("position_embedding", "learned"),
            "arch": cfg.get("arch", "trans_enc"),
            "freq_shift": cfg.get("freq_shift", 0),
            "time_encoded_dim": cfg.get("time_encoded_dim", 64),
            "s_audio_dim": cfg.get("s_audio_dim", 768),
            "s_audio_scale": cfg.get("s_audio_scale", cfg.get("latent_dim", 512) ** -0.5),
            "s_emotion_dim": cfg.get("s_emotion_dim", 25),
            "s_3dmm_dim": cfg.get("s_3dmm_dim", 58),
            "concat": cfg.get("concat", "concat_first"),
            "condition_concat": cfg.get("condition_concat", "token_concat"),
            "guidance_scale": cfg.get("guidance_scale", 7.5),
            "s_audio_enc_drop_prob": cfg.get("s_audio_enc_drop_prob", 0.2),
            "s_latent_embed_drop_prob": cfg.get("s_latent_embed_drop_prob", 0.2),
            "s_3dmm_enc_drop_prob": cfg.get("s_3dmm_enc_drop_prob", 0.2),
            "s_emotion_enc_drop_prob": cfg.get("s_emotion_enc_drop_prob", 1.0),
            "past_l_emotion_drop_prob": cfg.get("past_l_emotion_drop_prob", 1.0),
        }
        self.use_past_frames = cfg.get("use_past_frames", False)

        self.model = TransformerDenoiser(**self.init_params)

        self.decoder_diffusion = DecoderLatentDiffusion(
            conf.scheduler,
            conf.scheduler.num_train_timesteps,
            conf.scheduler.num_inference_timesteps,
        )
        self.schedule_sampler = UniformSampler(self.decoder_diffusion)
        self.num_preds = conf.scheduler.num_preds

    def _forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            past_listener_emotion=None,
            motion_length=None,
    ):
        with torch.no_grad():
            s_audio_encodings = self.audio_encoder._encode(speaker_audio_input)
            s_audio_encodings = s_audio_encodings.repeat_interleave(self.num_preds, dim=0)

          # freeze latent RNN_VAE embedder to extract speaker latent embedding
            s_latent_embed = self.latent_embedder.encode(speaker_emotion_input).unsqueeze(1)
            s_latent_embed = s_latent_embed.repeat_interleave(self.num_preds, dim=0)
            # shape: (batch_size * num_preds, 1, ...)

            # s_3dmm_encodings = self.latent_3dmm_embedder.get_encodings(speaker_3dmm_input)
            s_3dmm_encodings = speaker_3dmm_input.repeat_interleave(self.num_preds, dim=0)
            # shape: (bs * num_preds, s_w, ...)

            s_emotion_encodings = speaker_emotion_input.repeat_interleave(self.num_preds, dim=0)
            # shape: (bs * num_preds, s_w, ...)

            past_listener_emotion = past_listener_emotion.repeat_interleave(
                self.num_preds, dim=0) if past_listener_emotion is not None else None
            # shape: (bs * num_preds, l_w, ...)

            motion_length = motion_length.repeat_interleave(
                self.num_preds, dim=0) if motion_length is not None else None

            model_kwargs = {
                "speaker_audio_encodings": s_audio_encodings,
                "speaker_latent_embed": s_latent_embed,
                "speaker_3dmm_encodings": s_3dmm_encodings,
                "speaker_emotion_encodings": s_emotion_encodings,
                "past_listener_emotion": past_listener_emotion,
                "motion_length": motion_length,
            }

        if self.stage == "test":
            bs, l, _ = s_audio_encodings.shape  # bz * num_preds
            with torch.no_grad():
                output = [output for output in self.decoder_diffusion.ddim_sample_loop_progressive(
                    matcher=self,
                    model=self.model,
                    model_kwargs=model_kwargs,
                    shape=(bs, self.window_size if self.task == "online" else l, self.emotion_dim),
                )][-1]  # get last output

            output_listener_emotion = output["sample_enc"]  # (bz * num_preds, l_w, d=25)
            output_listener_emotion = rearrange(output_listener_emotion,
                                                "(b n) w d -> b n w d", n=self.num_preds)
            output_whole = {"prediction_emotion": output_listener_emotion}

        else:
            listener_emotion_input = listener_emotion_input.repeat_interleave(self.num_preds, dim=0)
            x_start_selected = listener_emotion_input  # (bs * num_preds, l_w, ...)

            t, _ = self.schedule_sampler.sample(x_start_selected.shape[0], x_start_selected.device)
            timesteps = t.long()

            output_whole = self.decoder_diffusion.denoise(self.model, x_start_selected, timesteps,
                                                          model_kwargs=model_kwargs)
            if motion_length is not None:  # offline task zero masking
                device = x_start_selected.get_device()
                output_mask = lengths_to_mask(motion_length, device=device, max_len=x_start_selected.shape[1])
                # print(f'output_whole["prediction_emotion"] shape: {output_whole["prediction_emotion"].shape}')
                output_whole["prediction_emotion"] = (output_whole["prediction_emotion"]
                                                      * output_mask.float().unsqueeze(-1))

            output_whole = {k: v.view(-1, self.num_preds, *output_whole[k].shape[1:]) for k, v in output_whole.items()}
        return output_whole

    def forward(self, **kwargs):
        return self._forward(**kwargs)


class LatentMatcher(nn.Module):
    def __init__(self,
                 task: str = "online",
                 stage: str = "fit",
                 device: str = None,
                 diffusion_prior: DictConfig = None,
                 diffusion_decoder: DictConfig = None,
                 latent_embedder: DictConfig = None,
                 audio_encoder: DictConfig = None,
                 eeg_head: DictConfig = None,
                 resumed_training: bool = False,
                 auto_load_ckpt: bool = True,
                 **kwargs):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.task = task
        self.stage = stage
        self.kwargs = kwargs

        module_dict_cfg = DictConfig(
            {"latent_embedder": latent_embedder,
             "audio_encoder": audio_encoder,}
        )

        self.diffusion_prior_cfg = diffusion_prior
        self.diffusion_prior = None
        if self.diffusion_prior_cfg is not None:
            self.diffusion_prior = PriorLatentMatcher(self.diffusion_prior_cfg,
                                                      task=task,
                                                      stage=stage,
                                                      module_dict_cfg=module_dict_cfg,
                                                      **kwargs)

        self.diffusion_decoder_cfg = diffusion_decoder
        self.diffusion_decoder = DecoderLatentMatcher(self.diffusion_decoder_cfg,
                                                      task=task,
                                                      stage=stage,
                                                      module_dict_cfg=module_dict_cfg,
                                                      **kwargs)
        self.eeg_head = None
        self.eeg_head_pooling = "mean"
        self.eeg_detach_prediction_emotion = True
        self.eeg_use_speaker_audio = True
        self.eeg_use_speaker_emotion = True
        self.eeg_use_speaker_3dmm = True
        self.eeg_use_prediction_emotion = True
        self.eeg_speaker_audio_dim = 0
        self.eeg_speaker_emotion_dim = 0
        self.eeg_speaker_3dmm_dim = 0
        self.eeg_prediction_emotion_dim = 0
        if eeg_head is not None and eeg_head.get("enabled", False):
            decoder_args = self.diffusion_decoder_cfg.args
            self.eeg_head_pooling = eeg_head.get("pooling", "mean")
            self.eeg_detach_prediction_emotion = eeg_head.get("detach_prediction_emotion", True)
            self.eeg_use_speaker_audio = eeg_head.get("use_speaker_audio", True)
            self.eeg_use_speaker_emotion = eeg_head.get("use_speaker_emotion", True)
            self.eeg_use_speaker_3dmm = eeg_head.get("use_speaker_3dmm", True)
            self.eeg_use_prediction_emotion = eeg_head.get("use_prediction_emotion", True)
            self.eeg_speaker_audio_dim = decoder_args.get("s_audio_dim", 768) \
                if self.eeg_use_speaker_audio else 0
            self.eeg_speaker_emotion_dim = decoder_args.get("s_emotion_dim", 25) \
                if self.eeg_use_speaker_emotion else 0
            self.eeg_speaker_3dmm_dim = decoder_args.get("s_3dmm_dim", 58) \
                if self.eeg_use_speaker_3dmm else 0
            self.eeg_prediction_emotion_dim = decoder_args.get("nfeats", 25) \
                if self.eeg_use_prediction_emotion else 0
            eeg_input_dim = (
                self.eeg_speaker_audio_dim
                + self.eeg_speaker_emotion_dim
                + self.eeg_speaker_3dmm_dim
                + self.eeg_prediction_emotion_dim
            )
            if eeg_input_dim <= 0:
                raise ValueError("At least one EEG head input source must be enabled.")
            self.eeg_head = EEGPredictionHead(
                input_dim=eeg_head.get("input_dim", eeg_input_dim),
                hidden_dim=eeg_head.get("hidden_dim", 256),
                output_dim=eeg_head.get("output_dim", 14),
                dropout=eeg_head.get("dropout", 0.5),
            )
        load_ckpt = False
        want_last = False
        want_best = False

        if resumed_training:
            load_ckpt = True
            want_last = True
        if stage == "test":
            load_ckpt = True
            want_best = True

        if load_ckpt and auto_load_ckpt:
            ckpt_path = self.get_ckpt_path(
                self.diffusion_decoder.model,
                runid="resume_runid",
                epoch=None,
                best=want_best,
                last=want_last,
            )
            from_pretrained_checkpoint(str(ckpt_path), self.diffusion_decoder.model, device)
            if self.diffusion_prior is not None:
                prior_ckpt_path = self.get_ckpt_path(
                    self.diffusion_prior.model,
                    runid="resume_runid",
                    epoch=None,
                    best=want_best,
                    last=want_last,
                    create_dir=False,
                )
                if os.path.exists(prior_ckpt_path):
                    from_pretrained_checkpoint(str(prior_ckpt_path), self.diffusion_prior.model, device)
                elif resumed_training:
                    raise FileNotFoundError(f"Missing prior checkpoint for resumed training: {prior_ckpt_path}")
            if self.eeg_head is not None:
                eeg_ckpt_path = self.get_ckpt_path(
                    self.eeg_head,
                    runid="resume_runid",
                    epoch=None,
                    best=want_best,
                    last=want_last,
                    create_dir=False,
                )
                if os.path.exists(eeg_ckpt_path):
                    from_pretrained_checkpoint(str(eeg_ckpt_path), self.eeg_head, device)
                elif resumed_training:
                    raise FileNotFoundError(f"Missing EEG head checkpoint for resumed training: {eeg_ckpt_path}")

    def freeze_except_eeg_head(self):
        if self.eeg_head is None:
            raise RuntimeError("Cannot train EEG head only because eeg_head is disabled.")

        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.eeg_head.parameters():
            parameter.requires_grad = True

    def set_eeg_head_train_mode(self):
        if self.eeg_head is None:
            raise RuntimeError("Cannot train EEG head only because eeg_head is disabled.")

        self.eval()
        self.eeg_head.train()

    def _pool_eeg_sequence(self, feature, expected_dim, batch_size, num_preds, device, dtype):
        if expected_dim <= 0:
            return None
        if feature is None or feature.numel() == 0:
            return torch.zeros(batch_size, num_preds, expected_dim, device=device, dtype=dtype)

        feature = feature.to(device=device, dtype=dtype)
        if feature.dim() == 3:
            if self.eeg_head_pooling == "last":
                pooled = feature[:, -1]
            elif self.eeg_head_pooling == "mean":
                pooled = feature.mean(dim=1)
            else:
                raise ValueError(f"Unknown EEG head pooling: {self.eeg_head_pooling}")
        elif feature.dim() == 2:
            pooled = feature
        else:
            raise ValueError(f"Unsupported EEG condition shape: {feature.shape}")

        return pooled.unsqueeze(1).expand(-1, num_preds, -1)

    def _attach_eeg_outputs(self, outputs, speaker_audio_input=None, speaker_emotion_input=None,
                            speaker_3dmm_input=None, listener_eeg_input=None, listener_eeg_mask=None):
        if self.eeg_head is None:
            return outputs

        prediction_emotion = outputs.get("prediction_emotion")
        if prediction_emotion is None:
            return outputs

        batch_size, num_preds = prediction_emotion.shape[:2]
        device = prediction_emotion.device
        dtype = prediction_emotion.dtype
        feature_list = []

        speaker_audio_feature = self._pool_eeg_sequence(
            speaker_audio_input, self.eeg_speaker_audio_dim, batch_size, num_preds, device, dtype)
        if speaker_audio_feature is not None:
            feature_list.append(speaker_audio_feature)

        speaker_emotion_feature = self._pool_eeg_sequence(
            speaker_emotion_input, self.eeg_speaker_emotion_dim, batch_size, num_preds, device, dtype)
        if speaker_emotion_feature is not None:
            feature_list.append(speaker_emotion_feature)

        speaker_3dmm_feature = self._pool_eeg_sequence(
            speaker_3dmm_input, self.eeg_speaker_3dmm_dim, batch_size, num_preds, device, dtype)
        if speaker_3dmm_feature is not None:
            feature_list.append(speaker_3dmm_feature)

        if self.eeg_head_pooling == "last":
            pooled_emotion = prediction_emotion[:, :, -1]
        elif self.eeg_head_pooling == "mean":
            pooled_emotion = prediction_emotion.mean(dim=2)
        else:
            raise ValueError(f"Unknown EEG head pooling: {self.eeg_head_pooling}")
        if self.eeg_detach_prediction_emotion:
            pooled_emotion = pooled_emotion.detach()
        if self.eeg_use_prediction_emotion:
            feature_list.append(pooled_emotion)

        prediction_eeg = self.eeg_head(torch.cat(feature_list, dim=-1))
        outputs["prediction_eeg"] = prediction_eeg

        if listener_eeg_input is None:
            return outputs

        target_eeg = listener_eeg_input.to(prediction_eeg.device).float()
        target_eeg_mask = listener_eeg_mask.to(prediction_eeg.device).float() \
            if listener_eeg_mask is not None else torch.ones_like(target_eeg)
        if target_eeg.dim() == 2:
            target_eeg = target_eeg.unsqueeze(1).expand(-1, num_preds, -1)
        if target_eeg_mask.dim() == 2:
            target_eeg_mask = target_eeg_mask.unsqueeze(1).expand(-1, num_preds, -1)
        outputs["target_eeg"] = target_eeg
        outputs["target_eeg_mask"] = target_eeg_mask
        return outputs

    def forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            listener_eeg_input=None,
            listener_eeg_mask=None,
            past_listener_emotion=None,
            motion_length=None,
    ):

        outputs = self.diffusion_decoder.forward(
            speaker_audio_input=speaker_audio_input,
            speaker_emotion_input=speaker_emotion_input,
            speaker_3dmm_input=speaker_3dmm_input,
            listener_emotion_input=listener_emotion_input,
            past_listener_emotion=past_listener_emotion,
            motion_length=motion_length,
        )
        outputs = self._attach_eeg_outputs(
            outputs,
            speaker_audio_input=speaker_audio_input,
            speaker_emotion_input=speaker_emotion_input,
            speaker_3dmm_input=speaker_3dmm_input,
            listener_eeg_input=listener_eeg_input,
            listener_eeg_mask=listener_eeg_mask,
        )
        # outputs['prediction_emotion']: (bz, num_preds, s_w, emotion_dim)
        if self.stage == "test" or self.diffusion_prior is None:
            return outputs

        output_prior = self.diffusion_prior.forward(
            speaker_audio_input=speaker_audio_input,
            speaker_emotion_input=speaker_emotion_input,
            speaker_3dmm_input=speaker_3dmm_input,
            listener_emotion_input=listener_emotion_input,
        )

        return {
            "output_prior": output_prior,
            "output_decoder": outputs,
        }

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

    def save_ckpt(self, optimizer, epoch=None, best=False, last=False, best_loss=float("inf")):
        models = [self.diffusion_decoder.model]
        if self.diffusion_prior is not None:
            models.append(self.diffusion_prior.model)
        if self.eeg_head is not None:
            models.append(self.eeg_head)

        for model in models:
            ckpt_path = self.get_ckpt_path(model, epoch=epoch, best=best, last=last)
            save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, best_loss=best_loss)

    def obtain_shapes(self, modified_layers):
        shape_dict = {}
        for name, module in self.named_modules():
            if name not in modified_layers:
                continue
            if hasattr(module, "weight"):
                shape_dict[name] = torch.tensor(module.weight.size())
            elif hasattr(module, "in_proj_weight"):
                shape_dict[name] = torch.tensor(module.in_proj_weight.size())
        return shape_dict
