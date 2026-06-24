import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2.0 / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2.0 / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2.0 / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        return torch.matmul(attn, value)


class PersonalFeatureEncoder(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=128, output_dim=512, k=2, dropout=0.1):
        super().__init__()
        clip_units = {1: 240, 2: 120, 3: 60, 4: 30}
        if k not in clip_units:
            raise ValueError(f"Unsupported PerReactor k value: {k}. Expected one of {sorted(clip_units)}.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = k
        self.clip_unit = clip_units[k]
        self.num_segments = 2 ** k
        self.required_len = self.clip_unit * self.num_segments

        self.listener_behaviour_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cross_attn = CrossAttn(in_channels=hidden_dim)
        self.temporal_layer = nn.Sequential(
            nn.LayerNorm(self.clip_unit * hidden_dim),
            nn.Linear(self.clip_unit * hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    def _prepare_clip(self, listener_personal_clip):
        if listener_personal_clip.dim() != 3:
            raise ValueError(
                "listener_personal_clip must have shape [batch, time, dim], "
                f"got {tuple(listener_personal_clip.shape)}"
            )
        if listener_personal_clip.shape[-1] != self.input_dim:
            raise ValueError(
                "listener_personal_clip feature dimension mismatch: "
                f"{listener_personal_clip.shape[-1]} != {self.input_dim}"
            )
        if listener_personal_clip.shape[1] >= self.required_len:
            return listener_personal_clip[:, :self.required_len]
        pad_len = self.required_len - listener_personal_clip.shape[1]
        pad = listener_personal_clip.new_zeros(
            listener_personal_clip.shape[0], pad_len, listener_personal_clip.shape[-1]
        )
        return torch.cat((listener_personal_clip, pad), dim=1)

    def forward(self, listener_personal_clip):
        encoded_feature = self.listener_behaviour_encoder(self._prepare_clip(listener_personal_clip))
        segments = [
            encoded_feature[:, idx * self.clip_unit:(idx + 1) * self.clip_unit]
            for idx in range(self.num_segments)
        ]
        while len(segments) > 1:
            half = len(segments) // 2
            segments = [self.cross_attn(segments[idx], segments[idx + half]) for idx in range(half)]

        personal_factor = self.temporal_layer(segments[0].reshape(segments[0].shape[0], -1))
        return self.output_layer(personal_factor)


def normalize_personal_condition_mode(mode):
    aliases = {
        "3dmm_only": "history_only",
        "3dmm_personality": "history_personality",
    }
    mode = aliases.get(mode, mode)
    valid_modes = {"history_only", "personality_only", "history_personality"}
    if mode not in valid_modes:
        raise ValueError(
            "Unsupported personal_condition_mode "
            f"'{mode}'. Expected one of {sorted(valid_modes | set(aliases.keys()))}."
        )
    return mode


class PersonalityEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, personality):
        if personality.dim() == 1:
            personality = personality.unsqueeze(0)
        if personality.dim() != 2:
            raise ValueError(f"Expected listener personality [batch, dim], got {tuple(personality.shape)}")
        return F.normalize(self.net(personality.float()), dim=-1)


class PersonalConditionFusion(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, history_embed, personality_embed):
        if history_embed.shape != personality_embed.shape:
            raise ValueError(
                "History and personality embeddings must have the same shape, got "
                f"{tuple(history_embed.shape)} and {tuple(personality_embed.shape)}"
            )
        return F.normalize(self.net(torch.cat([history_embed, personality_embed], dim=-1)), dim=-1)


class PerReactorOfflineModel(nn.Module):
    def __init__(
            self,
            diffusion,
            personal_feature_dim=25,
            personal_hidden_dim=128,
            personal_embed_dim=512,
            k=2,
            dropout=0.1,
            personal_condition_mode="history_only",
            personality_input_dim=5,
            personality_hidden_dim=128,
            personality_dropout=0.1,
            personality_fusion_hidden_dim=512,
            freeze_backbone=True,
    ):
        super().__init__()
        self.main_net = diffusion
        self.freeze_backbone_enabled = freeze_backbone
        self.personal_condition_mode = normalize_personal_condition_mode(personal_condition_mode)
        self.personal_encoder = PersonalFeatureEncoder(
            input_dim=personal_feature_dim,
            hidden_dim=personal_hidden_dim,
            output_dim=personal_embed_dim,
            k=k,
            dropout=dropout,
        )
        self.personality_encoder = None
        self.personality_fusion = None
        if self.uses_personality:
            self.personality_encoder = PersonalityEncoder(
                input_dim=personality_input_dim,
                hidden_dim=personality_hidden_dim,
                output_dim=personal_embed_dim,
                dropout=personality_dropout,
            )
        if self.uses_history and self.uses_personality:
            self.personality_fusion = PersonalConditionFusion(
                embed_dim=personal_embed_dim,
                hidden_dim=personality_fusion_hidden_dim,
                dropout=personality_dropout,
            )
        if self.freeze_backbone_enabled:
            self.freeze_backbone()

    @property
    def uses_history(self):
        return self.personal_condition_mode in {"history_only", "history_personality"}

    @property
    def uses_personality(self):
        return self.personal_condition_mode in {"personality_only", "history_personality"}

    def freeze_backbone(self):
        for parameter in self.main_net.parameters():
            parameter.requires_grad = False
        self.main_net.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone_enabled:
            self.main_net.eval()
        return self

    def eeg_head(self):
        if hasattr(self.main_net, "eeg_head"):
            return self.main_net.eeg_head
        denoise_fn = getattr(self.main_net, "denoise_fn", None)
        if denoise_fn is not None and hasattr(denoise_fn, "eeg_head"):
            return denoise_fn.eeg_head
        return None

    def has_eeg_head(self):
        return self.eeg_head() is not None

    def set_eeg_head_requires_grad(self, requires_grad=True):
        eeg_head = self.eeg_head()
        if eeg_head is None:
            raise ValueError("EEG head is not available in the generic backbone.")
        for parameter in eeg_head.parameters():
            parameter.requires_grad = requires_grad

    def freeze_except_eeg_head(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.set_eeg_head_requires_grad(True)
        self.main_net.eval()
        self.personal_encoder.eval()
        if self.personality_encoder is not None:
            self.personality_encoder.eval()
        if self.personality_fusion is not None:
            self.personality_fusion.eval()

    def set_eeg_head_train_mode(self):
        eeg_head = self.eeg_head()
        if eeg_head is None:
            raise ValueError("EEG head is not available in the generic backbone.")
        eeg_head.train(True)

    def adapter_parameters(self, include_eeg_head=False):
        parameters = list(self.personal_encoder.parameters())
        if self.personality_encoder is not None:
            parameters.extend(self.personality_encoder.parameters())
        if self.personality_fusion is not None:
            parameters.extend(self.personality_fusion.parameters())
        if include_eeg_head:
            eeg_head = self.eeg_head()
            if eeg_head is None:
                raise ValueError("Cannot train EEG head because the generic backbone has no EEG head.")
            parameters.extend(eeg_head.parameters())
        return parameters

    def adapter_state_dict(self, include_eeg_head=False):
        state = {f"personal_encoder.{name}": value for name, value in self.personal_encoder.state_dict().items()}
        if self.personality_encoder is not None:
            state.update(
                {f"personality_encoder.{name}": value for name, value in self.personality_encoder.state_dict().items()}
            )
        if self.personality_fusion is not None:
            state.update(
                {f"personality_fusion.{name}": value for name, value in self.personality_fusion.state_dict().items()}
            )
        if include_eeg_head:
            eeg_head = self.eeg_head()
            if eeg_head is None:
                raise ValueError("Cannot save EEG head because the generic backbone has no EEG head.")
            state.update({f"eeg_head.{name}": value for name, value in eeg_head.state_dict().items()})
        return state

    def load_adapter_state_dict(self, state_dict, strict=True):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing_keys = []
        unexpected_keys = []
        personal_state = {
            key[len("personal_encoder."):]: value
            for key, value in state_dict.items()
            if key.startswith("personal_encoder.")
        }
        if not personal_state:
            personal_state = {
                key: value for key, value in state_dict.items()
                if not key.startswith(("personality_encoder.", "personality_fusion.", "eeg_head."))
            }
        result = self.personal_encoder.load_state_dict(personal_state, strict=strict)
        missing_keys.extend([f"personal_encoder.{key}" for key in result.missing_keys])
        unexpected_keys.extend([f"personal_encoder.{key}" for key in result.unexpected_keys])

        if self.personality_encoder is not None:
            personality_state = {
                key[len("personality_encoder."):]: value
                for key, value in state_dict.items()
                if key.startswith("personality_encoder.")
            }
            personality_result = self.personality_encoder.load_state_dict(personality_state, strict=strict)
            missing_keys.extend([f"personality_encoder.{key}" for key in personality_result.missing_keys])
            unexpected_keys.extend([f"personality_encoder.{key}" for key in personality_result.unexpected_keys])

        if self.personality_fusion is not None:
            fusion_state = {
                key[len("personality_fusion."):]: value
                for key, value in state_dict.items()
                if key.startswith("personality_fusion.")
            }
            fusion_result = self.personality_fusion.load_state_dict(fusion_state, strict=strict)
            missing_keys.extend([f"personality_fusion.{key}" for key in fusion_result.missing_keys])
            unexpected_keys.extend([f"personality_fusion.{key}" for key in fusion_result.unexpected_keys])

        eeg_state = {
            key[len("eeg_head."):]: value
            for key, value in state_dict.items()
            if key.startswith("eeg_head.")
        }
        if eeg_state:
            eeg_head = self.eeg_head()
            if eeg_head is None:
                if strict:
                    raise ValueError("Checkpoint contains eeg_head weights, but the generic backbone has no EEG head.")
                unexpected_keys.extend([f"eeg_head.{key}" for key in eeg_state.keys()])
            else:
                eeg_result = eeg_head.load_state_dict(eeg_state, strict=strict)
                missing_keys.extend([f"eeg_head.{key}" for key in eeg_result.missing_keys])
                unexpected_keys.extend([f"eeg_head.{key}" for key in eeg_result.unexpected_keys])

        return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)

    def encode_listener_personal(self, listener_personal_clip):
        return self.personal_encoder(listener_personal_clip)

    def encode_person_condition(self, listener_personal_clip=None, personality=None):
        if self.uses_history:
            if listener_personal_clip is None or listener_personal_clip.numel() == 0:
                raise ValueError(
                    f"personal_condition_mode='{self.personal_condition_mode}' requires listener emotion history."
                )
            history_embed = self.personal_encoder(listener_personal_clip)
        else:
            history_embed = None

        if self.uses_personality:
            if personality is None or personality.numel() == 0:
                raise ValueError(
                    f"personal_condition_mode='{self.personal_condition_mode}' requires listener personality traits."
                )
            personality_embed = self.personality_encoder(personality)
        else:
            personality_embed = None

        if self.personal_condition_mode == "history_only":
            return history_embed
        if self.personal_condition_mode == "personality_only":
            return personality_embed
        return self.personality_fusion(history_embed, personality_embed)

    def forward(self, x, listener_personal_clip=None, listener_personal_embed=None, personality=None):
        if listener_personal_embed is None:
            listener_personal_embed = self.encode_person_condition(
                listener_personal_clip=listener_personal_clip,
                personality=personality,
            )
        return self.main_net(**x, listener_personal_embed=listener_personal_embed)
