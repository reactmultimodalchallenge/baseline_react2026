from collections import OrderedDict

import hydra
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.perfrdiff_rewrite_weight.person_specific.PersonSpecificEncoder import Transformer
from framework.utils.util import from_pretrained_checkpoint


def compute_regular_loss(weights):
    loss = weights[0].new_tensor(0.0)
    for weight in weights:
        loss = loss + torch.norm(weight.reshape(-1), 2)
    return loss


class ModifierNetwork(nn.Module):
    def __init__(self, input_dim=512, latent_dim=1024, output_dim=None, num_shared_layers=1):
        super().__init__()
        output_dim = output_dim or []
        self.shared_layers = nn.ModuleList(
            [
                nn.Linear(input_dim, latent_dim) if idx == 0 else nn.Linear(latent_dim, latent_dim)
                for idx in range(num_shared_layers)
            ]
        )
        self.output_dim = output_dim
        self.branches = nn.ModuleList(
            [nn.Linear(latent_dim, int(torch.prod(shape).item())) for shape in output_dim]
        )

    def forward(self, x):
        for layer in self.shared_layers:
            x = torch.relu(layer(x))
        return [
            branch(x).view([int(dim) for dim in self.output_dim[idx]])
            for idx, branch in enumerate(self.branches)
        ]

    def get_model_name(self):
        return self.__class__.__name__


class PersonalityEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class PersonalityFusion(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, history_embedding, personality_embedding):
        fused = self.net(torch.cat((history_embedding, personality_embedding), dim=-1))
        return F.normalize(fused, dim=-1)


class MainNetUnified(nn.Module):
    def __init__(self, cfg, main_net, device):
        super().__init__()
        self.main_net = main_net
        self.modified_layers = list(cfg.main_model.args.modified_layers)
        self.hypernet_predict = cfg.main_model.args.get("predict", "shift")
        self.crossattn_modify = cfg.main_model.args.get("modify", "all")
        self.regularization = cfg.main_model.args.get("regularization", False)
        self.regular_w = cfg.main_model.args.get("regular_w", 0.0)
        self.embed_dim = cfg.main_model.args.get("embed_dim", 512)
        self.personal_condition_mode = cfg.main_model.args.get("personal_condition_mode", "3dmm_personality")
        if self.personal_condition_mode not in {"3dmm_personality", "personality_only", "3dmm_only"}:
            raise ValueError(f"Unsupported personal_condition_mode: {self.personal_condition_mode}")

        for parameter in self.main_net.parameters():
            parameter.requires_grad = False

        modules = OrderedDict(self.main_net.named_modules())
        missing = [name for name in self.modified_layers if name not in modules]
        if missing:
            raise ValueError(f"Cannot find modified layers in diffusion model: {missing}")
        self.hooked_modules = OrderedDict((name, modules[name]) for name in self.modified_layers)

        weight_shapes = self.main_net.obtain_shapes(self.modified_layers)
        missing_shapes = [name for name in self.modified_layers if name not in weight_shapes]
        if missing_shapes:
            raise ValueError(f"Cannot infer weights for modified layers: {missing_shapes}")

        self.weight_shapes = []
        for layer_name in self.modified_layers:
            shape = weight_shapes[layer_name]
            if "multihead_attn" in layer_name and self.crossattn_modify == "kv":
                original_dim = shape[0]
                shape = torch.tensor([2 * torch.div(original_dim, 3, rounding_mode="trunc"), shape[1]])
            self.weight_shapes.append(shape)

        self.hypernet = ModifierNetwork(
            input_dim=cfg.main_model.args.get("input_dim", 512),
            latent_dim=cfg.main_model.args.get("latent_dim", 1024),
            output_dim=self.weight_shapes,
            num_shared_layers=cfg.main_model.args.get("num_shared_layers", 1),
        )

        self.personality_encoder = None
        self.personality_fusion = None
        if self.personal_condition_mode in {"3dmm_personality", "personality_only"}:
            personality_input_dim = cfg.main_model.args.get("personality_input_dim", 5)
            personality_hidden_dim = cfg.main_model.args.get("personality_hidden_dim", 128)
            personality_dropout = cfg.main_model.args.get("personality_dropout", 0.1)
            self.personality_encoder = PersonalityEncoder(
                input_dim=personality_input_dim,
                hidden_dim=personality_hidden_dim,
                output_dim=self.embed_dim,
                dropout=personality_dropout,
            )
            if self.personal_condition_mode == "3dmm_personality":
                fusion_hidden_dim = cfg.main_model.args.get("personality_fusion_hidden_dim", self.embed_dim)
                self.personality_fusion = PersonalityFusion(
                    embed_dim=self.embed_dim,
                    hidden_dim=fusion_hidden_dim,
                    dropout=personality_dropout,
                )

        self.person_encoder = None
        if self.personal_condition_mode in {"3dmm_personality", "3dmm_only"}:
            person_cfg = cfg.person_specific
            self.person_encoder = Transformer(device, **person_cfg.args)
            checkpoint_path = hydra.utils.to_absolute_path(person_cfg.checkpoint_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Missing person-specific encoder checkpoint: {checkpoint_path}. "
                    "Please place it under pretrained_models/person_specific/ or override "
                    "trainer.person_specific.checkpoint_path."
                )
            from_pretrained_checkpoint(checkpoint_path, self.person_encoder, device)
            self.person_encoder.eval()
            for parameter in self.person_encoder.parameters():
                parameter.requires_grad = False

        self._initialize_editable_weights(device)

    def _initialize_editable_weights(self, device):
        original_weights = []
        weight_kinds = []
        for name, module in self.hooked_modules.items():
            if hasattr(module, "weight"):
                original_weights.append(module.weight.detach())
                weight_kinds.append("weight")
                del module._parameters["weight"]
            elif hasattr(module, "in_proj_weight"):
                original_weights.append(module.in_proj_weight.detach())
                weight_kinds.append("in_proj_weight")
                del module._parameters["in_proj_weight"]
            else:
                raise ValueError(f"Layer has no editable weight: {name}")
        self.original_weights = original_weights
        self.weight_kinds = weight_kinds

        self.tensor_0 = torch.zeros(size=(self.embed_dim, self.embed_dim), device=device)
        self.tensor_1 = torch.tensor(1.0, device=device)

    def eeg_head(self):
        return getattr(self.main_net, "eeg_head", None)

    def has_eeg_head(self):
        return self.eeg_head() is not None

    def set_eeg_head_requires_grad(self, requires_grad=True):
        if not self.has_eeg_head():
            raise RuntimeError("Cannot train/evaluate EEG because main_net.eeg_head is disabled.")
        for parameter in self.eeg_head().parameters():
            parameter.requires_grad = requires_grad

    def freeze_except_eeg_head(self):
        if not self.has_eeg_head():
            raise RuntimeError("Cannot train EEG head only because main_net.eeg_head is disabled.")
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.set_eeg_head_requires_grad(True)

    def set_eeg_head_train_mode(self):
        if not self.has_eeg_head():
            raise RuntimeError("Cannot train EEG head only because main_net.eeg_head is disabled.")
        self.eval()
        self.eeg_head().train()

    def modifier_parameters(self, include_eeg_head=False):
        for parameter in self.hypernet.parameters():
            yield parameter
        if self.personality_encoder is not None:
            for parameter in self.personality_encoder.parameters():
                yield parameter
        if self.personality_fusion is not None:
            for parameter in self.personality_fusion.parameters():
                yield parameter
        if include_eeg_head:
            self.set_eeg_head_requires_grad(True)
            for parameter in self.eeg_head().parameters():
                yield parameter

    def modifier_state_dict(self, include_eeg_head=False):
        state_dict = {
            f"hypernet.{name}": value
            for name, value in self.hypernet.state_dict().items()
        }
        if self.personality_encoder is not None:
            state_dict.update(
                {
                    f"personality_encoder.{name}": value
                    for name, value in self.personality_encoder.state_dict().items()
                }
            )
        if self.personality_fusion is not None:
            state_dict.update(
                {
                    f"personality_fusion.{name}": value
                    for name, value in self.personality_fusion.state_dict().items()
                }
            )
        if include_eeg_head and self.has_eeg_head():
            state_dict.update(
                {
                    f"eeg_head.{name}": value
                    for name, value in self.eeg_head().state_dict().items()
                }
            )
        return state_dict

    def load_modifier_state_dict(self, state_dict):
        if any(key.startswith("hypernet.") for key in state_dict):
            hypernet_state = {
                key[len("hypernet."):]: value
                for key, value in state_dict.items()
                if key.startswith("hypernet.")
            }
            self.hypernet.load_state_dict(hypernet_state)

            personality_state = {
                key[len("personality_encoder."):]: value
                for key, value in state_dict.items()
                if key.startswith("personality_encoder.")
            }
            if personality_state:
                if self.personality_encoder is None:
                    raise ValueError(
                        "Checkpoint contains personality_encoder weights, "
                        f"but current personal_condition_mode is {self.personal_condition_mode}."
                    )
                self.personality_encoder.load_state_dict(personality_state)

            if self.personality_fusion is not None:
                fusion_state = {
                    key[len("personality_fusion."):]: value
                    for key, value in state_dict.items()
                    if key.startswith("personality_fusion.")
                }
                if fusion_state:
                    self.personality_fusion.load_state_dict(fusion_state)
            eeg_state = {
                key[len("eeg_head."):]: value
                for key, value in state_dict.items()
                if key.startswith("eeg_head.")
            }
            if eeg_state:
                if not self.has_eeg_head():
                    raise ValueError("Checkpoint contains eeg_head weights, but main_net.eeg_head is disabled.")
                self.eeg_head().load_state_dict(eeg_state)
            return

        self.hypernet.load_state_dict(state_dict)

    def encode_person_condition(self, p=None, personality=None):
        if self.personal_condition_mode == "3dmm_only":
            if p is None or p.numel() == 0:
                raise ValueError("3dmm_only mode requires a non-empty personal 3DMM history.")
            with torch.no_grad():
                _, person_embedding = self.person_encoder(p)
        else:
            if personality is None or personality.numel() == 0:
                raise ValueError("listener personality is required for perfrdiff personal conditioning.")
            if personality.dim() == 1:
                personality = personality.unsqueeze(0)
            personality_embedding = self.personality_encoder(personality.float())
            if self.personal_condition_mode == "personality_only":
                person_embedding = personality_embedding
            else:
                if p is None or p.numel() == 0:
                    raise ValueError("3dmm_personality mode requires a non-empty personal 3DMM history.")
                with torch.no_grad():
                    _, history_embedding = self.person_encoder(p)
                person_embedding = self.personality_fusion(history_embedding, personality_embedding)

        if person_embedding.shape[0] != 1:
            raise ValueError(
                "ModifierNetwork expects exactly one personal reference per forward. "
                "Use trainer-side micro-forwarding for batch training."
            )
        return person_embedding

    def apply_weights(self):
        for idx, (name, module) in enumerate(self.hooked_modules.items()):
            delta_w = self.kernel[idx]
            if self.weight_kinds[idx] == "weight":
                if self.hypernet_predict == "shift":
                    module.weight = self.original_weights[idx] + delta_w
                elif self.hypernet_predict == "offset":
                    module.weight = self.original_weights[idx] * (self.tensor_1 + delta_w)
                elif self.hypernet_predict == "weight":
                    module.weight = delta_w
                else:
                    raise ValueError(f"Unsupported hypernet prediction mode: {self.hypernet_predict}")
            else:
                if "multihead_attn" in name and self.crossattn_modify == "kv":
                    delta_w = torch.cat((self.tensor_0, delta_w), dim=0)
                if self.hypernet_predict == "shift":
                    module.in_proj_weight = self.original_weights[idx] + delta_w
                elif self.hypernet_predict == "offset":
                    module.in_proj_weight = self.original_weights[idx] * (self.tensor_1 + delta_w)
                elif self.hypernet_predict == "weight":
                    module.in_proj_weight = delta_w
                else:
                    raise ValueError(f"Unsupported hypernet prediction mode: {self.hypernet_predict}")

    def forward(self, x, p=None, personality=None):
        person_embedding = self.encode_person_condition(p=p, personality=personality)
        self.kernel = self.hypernet(person_embedding)
        self.apply_weights()
        output = self.main_net(**x)
        if self.regularization:
            return output, self.regular_w * compute_regular_loss(self.kernel)
        return output, person_embedding.new_tensor(0.0)
