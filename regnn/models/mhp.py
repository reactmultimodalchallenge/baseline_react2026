import torch
import torch.nn as nn
from utils.compute_distance_fun import compute_distance


NUM_SAMPLE=10


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


class MHP(nn.Module):
    def __init__(self, p=None, c=None, m=None, no_inverse=False, dist="MSE", neighbor_pattern='all',
                 eeg_head_enabled=False, eeg_input_dim=1586, eeg_hidden_dim=256, eeg_output_dim=14,
                 eeg_dropout=0.5, eeg_pooling='mean', eeg_detach_prediction_emotion=True):
        super().__init__()
        self.perceptual_processor = p
        self.cognitive_processor = c
        self.motor_processor = m
        self.cal_dist = {
            "DTW": compute_distance,
            "MSE": nn.functional.mse_loss,
        }
        self.no_inverse = no_inverse
        self.neighbor_pattern = neighbor_pattern
        self.eeg_head = None
        self.eeg_input_dim = eeg_input_dim
        self.eeg_head_pooling = eeg_pooling
        self.eeg_detach_prediction_emotion = eeg_detach_prediction_emotion
        if eeg_head_enabled:
            self.eeg_head = EEGPredictionHead(
                input_dim=eeg_input_dim,
                hidden_dim=eeg_hidden_dim,
                output_dim=eeg_output_dim,
                dropout=eeg_dropout,
            )

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

    def _pool_eeg_feature(self, feature):
        if feature is None:
            return None
        if feature.dim() == 1:
            return feature.unsqueeze(0)
        if feature.dim() == 2:
            return feature
        if feature.dim() == 3:
            if self.eeg_head_pooling == 'last':
                return feature[:, -1]
            if self.eeg_head_pooling == 'mean':
                return feature.mean(dim=1)
            raise ValueError(f"Unknown EEG head pooling: {self.eeg_head_pooling}")
        raise ValueError(f"Unsupported EEG feature shape: {feature.shape}")

    def predict_eeg(self, video_inputs, audio_inputs, speaker_emotion_inputs, prediction_emotion):
        if self.eeg_head is None:
            raise RuntimeError("Cannot predict EEG because eeg_head is disabled.")

        prediction_emotion_feature = self._pool_eeg_feature(prediction_emotion)
        if self.eeg_detach_prediction_emotion:
            prediction_emotion_feature = prediction_emotion_feature.detach()

        feature_list = [
            self._pool_eeg_feature(video_inputs),
            self._pool_eeg_feature(audio_inputs),
            self._pool_eeg_feature(speaker_emotion_inputs),
            prediction_emotion_feature,
        ]
        feature_list = [feature for feature in feature_list if feature is not None]
        eeg_feature = torch.cat(feature_list, dim=-1)
        if eeg_feature.shape[-1] != self.eeg_input_dim:
            raise RuntimeError(
                f"EEG head input dim mismatch: got {eeg_feature.shape[-1]}, "
                f"expected {self.eeg_input_dim}. Set --eeg-input-dim to match the "
                "concatenated feature size."
            )
        return self.eeg_head(eeg_feature)

    def forward_features(self, video_inputs, audio_inputs):
        if video_inputs is None:
            fused_features = audio_inputs
        elif audio_inputs is None:
            fused_features = video_inputs
        else:
            # B, T, 64
            fused_features = self.perceptual_processor(video_inputs, audio_inputs)
        # cog_outputs: 4, 25, 50 ---> B, N, T
        # edge: [4, 8, 25, 25]
        cog_outputs, edge, params = self.cognitive_processor(fused_features)
        return cog_outputs, edge, params

    def forward(self, video_inputs, audio_inputs, targets, lengthes=None):
        # speaker_feature: 4, 25, 50
        speaker_feature, edge, params = self.forward_features(video_inputs, audio_inputs)
        if not self.no_inverse:
            edge = torch.repeat_interleave(edge, repeats=torch.tensor(lengthes, device=edge.device), dim=0)
            self.motor_processor.train()
            # Encode all appropriate real facial reactions to a GMGD distribution
            # listener_feature: B, N, D
            listener_feature, logdets = self.motor_processor(targets, edge)

            return speaker_feature, listener_feature, params, edge, targets, logdets

        else:
            # Decode samples to listener appropriate facial reactions
            listerer_feature, logdets = self.motor_processor(speaker_feature, edge)
            nearest_targets = targets

            return listerer_feature, nearest_targets, logdets

    def inverse(self, video_inputs, audio_inputs, cal_norm, threshold=None):
        """ test stage """
        speaker_feature, edge, params = self.forward_features(video_inputs, audio_inputs)
        if not self.no_inverse:
            speaker_feature = self.sample(speaker_feature, threshold)
            predictions = self.motor_processor.inverse(speaker_feature, edge=edge, cal_norm=cal_norm)
        else:
            speaker_feature = self.sample(speaker_feature, threshold)
            predictions, _ = self.motor_processor(speaker_feature, edge)
        return predictions.transpose(2, 1)

    def sample(self, speaker_feature, threshold=None):
        noise = torch.randn(speaker_feature.shape, device=speaker_feature.device)
        if threshold is None:
            return speaker_feature + noise
        threshold = torch.sqrt(torch.tensor([threshold], device=speaker_feature.device))

        max_abs = torch.max(torch.abs(noise))
        if max_abs <= threshold:
            return speaker_feature + noise
        scale = threshold / max_abs
        scaled_noise = noise * scale
        return speaker_feature + scaled_noise

    
# def get_nearest(self, features, edge, targets):
#     # B, 750, 25
#     with torch.no_grad():
#         if not self.no_inverse:
#             self.motor_processor.eval()
#             predictions = self.motor_processor.inverse(features, edge=edge)
#         else:
#             predictions = features
#         B = predictions.shape[0]
#         nearest_idx = []
#         for i in range(B):
#             if len(targets[1]) == None:
#                 nearest_idx.append(0)
#                 continue
# 
#             pred = predictions[i].unsqueeze(0)
#             pair_targets = targets[i]
#             min_dist = None
#             for i, pair_target in enumerate(pair_targets):
#                 if pair_target == None:
#                     continue
#                 dist = self.cal_dist(pred, pair_target.transpose(1, 0).unsqueeze(0))
#                 if min_dist == None or dist < min_dist:
#                     min_dist = dist
#                     min_inx = i
# 
#             nearest_idx.append(min_inx)
# 
#     nearest_targets = [targets[i][idx].transpose(1, 0) for i, idx in enumerate(nearest_idx)]
#     nearest_targets = torch.stack(nearest_targets, dim=0)
#     return nearest_targets


# def onlyInverseMotor(self, features, edge):
#     print('='*50)
#     print('--------------------Only Inverse--------------------')
#     with torch.no_grad():
#         self.motor_processor.eval()
#         outputs = self.motor_processor.inverse(features, edge=edge)
#
#     return outputs
