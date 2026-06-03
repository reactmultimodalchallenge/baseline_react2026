import torch
import torch.nn as nn
from torch import Tensor
from .BasicBlock import ConvBlock, PositionalEncoding, init_biased_mask


def lengths_to_mask(lengths,
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


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


class VideoEncoder(nn.Module):
    def __init__(self, img_size=224, feature_dim=128, device='cpu'):
        super(VideoEncoder, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim

        self.Conv3D = ConvBlock(3, feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.device = device

    def forward(self, video):
        """
        input:
        speaker_video_frames x: (batch_size, seq_len, 3, img_size, img_size)

        output:
        speaker_temporal_tokens y: (batch_size, seq_len, token_dim)

        """

        video_input = video.transpose(1, 2)  # B C T H W
        token_output = self.Conv3D(video_input).transpose(1, 2)
        token_output = self.fc(token_output)  # B T C
        return token_output


class VAEModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 256,
                 **kwargs) -> None:
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.1)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))

    def forward(self, input):
        x = self.linear(input)  # B T D
        B, T, D = input.shape

        lengths = [len(item) for item in input]

        mu_token = torch.tile(self.mu_token, (B,)).reshape(B, 1, -1)
        logvar_token = torch.tile(self.logvar_token, (B,)).reshape(B, 1, -1)

        x = torch.cat([mu_token, logvar_token, x], dim=1)

        x = x.permute(1, 0, 2)

        token_mask = torch.ones((B, 2), dtype=bool, device=input.get_device())
        mask = lengths_to_mask(lengths, input.get_device())

        aug_mask = torch.cat((token_mask, mask), 1)

        x = self.seqTransEncoder(x, src_key_padding_mask=~aug_mask)

        mu = x[0]
        logvar = x[1]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        motion_sample = self.sample_from_distribution(dist).to(input.get_device())
        return motion_sample, dist

    def sample_from_distribution(self, distribution):
        return distribution.rsample()


class Decoder(nn.Module):
    def __init__(self, output_3dmm_dim=58, output_emotion_dim=25, feature_dim=128, device='cpu', max_seq_len=751,
                 n_head=4, window_size=8, online=False):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.device = device
        self.online = online

        self.vae_model = VAEModel(feature_dim, feature_dim)

        if self.online:
            self.lstm = nn.LSTM(feature_dim, feature_dim, 1, batch_first=True)
            self.linear_3d = nn.Linear(output_3dmm_dim, feature_dim)
            self.linear_reaction = nn.Linear(feature_dim, feature_dim)
            decoder_layer_3d = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=2 * feature_dim,
                                                          batch_first=True)
            self.listener_reaction_decoder_3d = nn.TransformerDecoder(decoder_layer_3d, num_layers=1)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2 * feature_dim,
                                                   batch_first=True)
        self.listener_reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.listener_reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.biased_mask = init_biased_mask(n_head=n_head, max_seq_len=max_seq_len, period=max_seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
            nn.Linear(feature_dim, output_emotion_dim)
        )
        self.PE = PositionalEncoding(feature_dim)

    def forward(self, encoded_feature, past_reaction_3dmm=None, past_reaction_emotion=None):
        B, TS = encoded_feature.shape[0], encoded_feature.shape[1]
        if self.online:
            TL = self.window_size
        else:
            TL = TS
        motion_sample, dist = self.vae_model(encoded_feature)
        time_queries = torch.zeros(B, TL, self.feature_dim, device=encoded_feature.get_device())
        time_queries = self.PE(time_queries)
        tgt_mask = self.biased_mask[:, :TL, :TL].clone().detach().to(device=self.device).repeat(B, 1, 1)

        listener_reaction = self.listener_reaction_decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1),
                                                             tgt_mask=tgt_mask)
        listener_reaction = self.listener_reaction_decoder_2(listener_reaction, listener_reaction, tgt_mask=tgt_mask)

        if self.online and (past_reaction_3dmm is not None):
            past_reaction_3dmm = self.linear_3d(past_reaction_3dmm)
            past_reaction_3dmm_last = past_reaction_3dmm[:, -1]

            tgt_mask = self.biased_mask[:, :(TL + past_reaction_3dmm.shape[1]),
                       :(TL + past_reaction_3dmm.shape[1])].detach().to(device=self.device).repeat(B, 1, 1)
            all_3dmm = torch.cat((past_reaction_3dmm, self.linear_reaction(listener_reaction)), dim=1)
            listener_3dmm_out = self.listener_reaction_decoder_3d(all_3dmm, all_3dmm, tgt_mask=tgt_mask)
            frame_num = listener_3dmm_out.shape[1]
            listener_3dmm_out = listener_3dmm_out[:, (frame_num - TL):]

            listener_3dmm_out, _ = self.lstm(listener_3dmm_out,
                                             (past_reaction_3dmm_last.view(1, B, self.feature_dim).contiguous(),
                                              past_reaction_3dmm_last.view(1, B, self.feature_dim).contiguous()))

            listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_3dmm_out)
        else:
            listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)

        listener_emotion_out = self.listener_reaction_emotion_map_layer(
            torch.cat((listener_3dmm_out, listener_reaction), dim=-1))

        return listener_3dmm_out, listener_emotion_out, dist

    def reset_window_size(self, window_size):
        self.window_size = window_size


class SpeakerBehaviourEncoder(nn.Module):
    def __init__(self, img_size=224, audio_dim=78, feature_dim=128, device='cpu'):
        super(SpeakerBehaviourEncoder, self).__init__()

        self.img_size = img_size
        self.audio_dim = audio_dim
        self.feature_dim = feature_dim
        self.device = device

        self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)
        self.audio_feature_map = nn.Linear(self.audio_dim, self.feature_dim)
        self.fusion_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)

    def forward(self, video, audio):
        video_feature = self.video_encoder(video)
        audio_feature = self.audio_feature_map(audio)
        speaker_behaviour_feature = self.fusion_layer(torch.cat((video_feature, audio_feature), dim=-1))

        return speaker_behaviour_feature


class TransformerVAE(nn.Module):
    def __init__(self, img_size=224, audio_dim=78, output_3dmm_dim=58, output_emotion_dim=25, feature_dim=128,
                 seq_len=750, task='online', window_size=8, device='cuda', eeg_head=None, **kwargs):
        super(TransformerVAE, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.online = True if task == 'online' else False
        self.window_size = window_size
        self.device = device

        self.speaker_behaviour_encoder = SpeakerBehaviourEncoder(img_size, audio_dim, feature_dim, device)
        self.reaction_decoder = Decoder(output_3dmm_dim=output_3dmm_dim, output_emotion_dim=output_emotion_dim,
                                        feature_dim=feature_dim, device=device, window_size=self.window_size,
                                        online=self.online)
        self.fusion = nn.Linear(feature_dim + self.output_3dmm_dim + self.output_emotion_dim, feature_dim)
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
            self.eeg_head_pooling = eeg_head.get("pooling", "mean")
            self.eeg_detach_prediction_emotion = eeg_head.get("detach_prediction_emotion", True)
            self.eeg_use_speaker_audio = eeg_head.get("use_speaker_audio", True)
            self.eeg_use_speaker_emotion = eeg_head.get("use_speaker_emotion", True)
            self.eeg_use_speaker_3dmm = eeg_head.get("use_speaker_3dmm", True)
            self.eeg_use_prediction_emotion = eeg_head.get("use_prediction_emotion", True)
            self.eeg_speaker_audio_dim = audio_dim if self.eeg_use_speaker_audio else 0
            self.eeg_speaker_emotion_dim = output_emotion_dim if self.eeg_use_speaker_emotion else 0
            self.eeg_speaker_3dmm_dim = output_3dmm_dim if self.eeg_use_speaker_3dmm else 0
            self.eeg_prediction_emotion_dim = output_emotion_dim if self.eeg_use_prediction_emotion else 0
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

    @staticmethod
    def _lengths_to_list(motion_lengths, batch_size, fallback_length):
        if motion_lengths is None:
            return [fallback_length] * batch_size
        if torch.is_tensor(motion_lengths):
            values = motion_lengths.detach().cpu().flatten().tolist()
        elif isinstance(motion_lengths, (list, tuple)):
            values = [
                int(item.item() if torch.is_tensor(item) else item)
                for item in motion_lengths
            ]
        else:
            values = [int(motion_lengths)]
        if len(values) == 1 and batch_size > 1:
            values = values * batch_size
        if len(values) < batch_size:
            values = values + [fallback_length] * (batch_size - len(values))
        return [int(value) for value in values[:batch_size]]

    def _pool_one_eeg_feature(self, feature, expected_dim, device, dtype):
        if feature is None or feature.numel() == 0:
            return torch.zeros(expected_dim, device=device, dtype=dtype)

        feature = feature.to(device=device, dtype=dtype)
        if feature.dim() == 1:
            return feature
        if feature.dim() == 2:
            if self.eeg_head_pooling == "last":
                return feature[-1]
            if self.eeg_head_pooling == "mean":
                return feature.mean(dim=0)
            raise ValueError(f"Unknown EEG head pooling: {self.eeg_head_pooling}")
        raise ValueError(f"Unsupported EEG feature shape: {feature.shape}")

    def _pool_eeg_sequence(self, feature, expected_dim, batch_size, device, dtype):
        if expected_dim <= 0:
            return None
        if feature is None:
            return torch.zeros(batch_size, expected_dim, device=device, dtype=dtype)

        if isinstance(feature, (list, tuple)):
            pooled = [
                self._pool_one_eeg_feature(item, expected_dim, device, dtype)
                for item in feature
            ]
            if len(pooled) == 0:
                return torch.zeros(batch_size, expected_dim, device=device, dtype=dtype)
            if len(pooled) < batch_size:
                pooled.extend(
                    torch.zeros(expected_dim, device=device, dtype=dtype)
                    for _ in range(batch_size - len(pooled))
                )
            return torch.stack(pooled[:batch_size], dim=0)

        feature = feature.to(device=device, dtype=dtype)
        if feature.numel() == 0:
            return torch.zeros(batch_size, expected_dim, device=device, dtype=dtype)
        if feature.dim() == 3:
            if self.eeg_head_pooling == "last":
                pooled = feature[:, -1]
            elif self.eeg_head_pooling == "mean":
                pooled = feature.mean(dim=1)
            else:
                raise ValueError(f"Unknown EEG head pooling: {self.eeg_head_pooling}")
        elif feature.dim() == 2:
            if feature.shape[0] == batch_size and feature.shape[-1] == expected_dim:
                pooled = feature
            else:
                pooled = self._pool_one_eeg_feature(feature, expected_dim, device, dtype).unsqueeze(0)
        elif feature.dim() == 1:
            pooled = feature.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported EEG condition shape: {feature.shape}")

        if pooled.shape[0] == 1 and batch_size > 1:
            pooled = pooled.expand(batch_size, -1)
        return pooled

    @staticmethod
    def _last_eeg_value(sequence, mask, length, device, dtype):
        if sequence is None or sequence.numel() == 0:
            return None, None

        sequence = sequence.to(device=device, dtype=dtype)
        if sequence.dim() == 1:
            target = sequence
            default_mask = torch.ones_like(target)
        elif sequence.dim() == 2:
            index = min(max(int(length) - 1, 0), sequence.shape[0] - 1)
            target = sequence[index]
            default_mask = torch.ones_like(target)
        else:
            raise ValueError(f"Unsupported EEG target shape: {sequence.shape}")

        if mask is None or mask.numel() == 0:
            return target, default_mask
        mask = mask.to(device=device, dtype=dtype)
        if mask.dim() == 1:
            return target, mask
        if mask.dim() == 2:
            index = min(max(int(length) - 1, 0), mask.shape[0] - 1)
            return target, mask[index]
        raise ValueError(f"Unsupported EEG mask shape: {mask.shape}")

    def _eeg_targets_from_sequences(self, listener_eeg_input, listener_eeg_mask,
                                    motion_lengths, batch_size, device, dtype):
        if listener_eeg_input is None:
            return None, None

        fallback_length = self.seq_len
        lengths = self._lengths_to_list(motion_lengths, batch_size, fallback_length)
        targets = []
        masks = []

        if isinstance(listener_eeg_input, (list, tuple)):
            mask_items = listener_eeg_mask if isinstance(listener_eeg_mask, (list, tuple)) else [None] * len(listener_eeg_input)
            for index, sequence in enumerate(listener_eeg_input[:batch_size]):
                mask = mask_items[index] if index < len(mask_items) else None
                target, target_mask = self._last_eeg_value(sequence, mask, lengths[index], device, dtype)
                if target is None:
                    continue
                targets.append(target)
                masks.append(target_mask)
        else:
            listener_eeg_input = listener_eeg_input.to(device=device, dtype=dtype)
            if listener_eeg_input.dim() == 3:
                listener_eeg_mask = listener_eeg_mask.to(device=device, dtype=dtype) \
                    if listener_eeg_mask is not None else None
                for index in range(min(batch_size, listener_eeg_input.shape[0])):
                    mask = listener_eeg_mask[index] if listener_eeg_mask is not None else None
                    target, target_mask = self._last_eeg_value(
                        listener_eeg_input[index], mask, lengths[index], device, dtype)
                    targets.append(target)
                    masks.append(target_mask)
            else:
                target, target_mask = self._last_eeg_value(
                    listener_eeg_input,
                    listener_eeg_mask.to(device=device, dtype=dtype) if listener_eeg_mask is not None else None,
                    lengths[0],
                    device,
                    dtype,
                )
                if target is not None:
                    targets.append(target)
                    masks.append(target_mask)

        if not targets:
            return None, None
        target_eeg = torch.stack(targets, dim=0)
        target_eeg_mask = torch.stack(masks, dim=0)
        if target_eeg.shape[0] == 1 and batch_size > 1:
            target_eeg = target_eeg.expand(batch_size, -1)
            target_eeg_mask = target_eeg_mask.expand(batch_size, -1)
        return target_eeg, target_eeg_mask

    def _attach_eeg_outputs(self, listener_emotion_out, speaker_audio=None, speaker_emotion=None,
                            speaker_3dmm=None, listener_eeg_input=None, listener_eeg_mask=None,
                            motion_lengths=None):
        if self.eeg_head is None:
            return {}
        if listener_emotion_out is None:
            return {}

        batch_size = len(listener_emotion_out) if isinstance(listener_emotion_out, list) else listener_emotion_out.shape[0]
        first_prediction = listener_emotion_out[0] if isinstance(listener_emotion_out, list) else listener_emotion_out
        device = first_prediction.device
        dtype = first_prediction.dtype
        feature_list = []

        speaker_audio_feature = self._pool_eeg_sequence(
            speaker_audio, self.eeg_speaker_audio_dim, batch_size, device, dtype)
        if speaker_audio_feature is not None:
            feature_list.append(speaker_audio_feature)

        speaker_emotion_feature = self._pool_eeg_sequence(
            speaker_emotion, self.eeg_speaker_emotion_dim, batch_size, device, dtype)
        if speaker_emotion_feature is not None:
            feature_list.append(speaker_emotion_feature)

        speaker_3dmm_feature = self._pool_eeg_sequence(
            speaker_3dmm, self.eeg_speaker_3dmm_dim, batch_size, device, dtype)
        if speaker_3dmm_feature is not None:
            feature_list.append(speaker_3dmm_feature)

        prediction_emotion_feature = self._pool_eeg_sequence(
            listener_emotion_out, self.eeg_prediction_emotion_dim, batch_size, device, dtype)
        if prediction_emotion_feature is not None:
            if self.eeg_detach_prediction_emotion:
                prediction_emotion_feature = prediction_emotion_feature.detach()
            feature_list.append(prediction_emotion_feature)

        prediction_eeg = self.eeg_head(torch.cat(feature_list, dim=-1))
        outputs = {"prediction_eeg": prediction_eeg}

        target_eeg, target_eeg_mask = self._eeg_targets_from_sequences(
            listener_eeg_input,
            listener_eeg_mask,
            motion_lengths,
            batch_size,
            prediction_eeg.device,
            prediction_eeg.dtype,
        )
        if target_eeg is not None:
            outputs["target_eeg"] = target_eeg
            outputs["target_eeg_mask"] = target_eeg_mask
        return outputs

    def forward(self, speaker_video=None, speaker_audio=None, **kwargs):
        speaker_emotion = kwargs.get("speaker_emotion", None)
        speaker_3dmm = kwargs.get("speaker_3dmm", None)
        listener_eeg_input = kwargs.get("listener_eeg_input", None)
        listener_eeg_mask = kwargs.get("listener_eeg_mask", None)
        return_eeg_outputs = kwargs.get("return_eeg_outputs", False)
        return_distribution = kwargs.get("return_distribution", True)
        distribution = [] if return_distribution else None
        if self.online:
            speaker_video = torch.stack(speaker_video, dim=0).to(device=self.device)
            speaker_audio = torch.stack(speaker_audio, dim=0).to(device=self.device)
            motion_lengths = (
                torch.as_tensor(
                    kwargs.get('motion_lengths', [self.seq_len] * len(speaker_video)), device=self.device,
                ).clamp(max=self.seq_len)
            )

            frame_num = speaker_video.shape[1]
            period = frame_num // self.window_size
            # num_windows = motion_lengths // self.window_size
            # motion_lengths Tensor([58, 720, 625, 750, ...]) ==> num_windows Tensor([7, 90, 78, 93, ...])

            reaction_3dmm = torch.zeros((speaker_video.size(0), self.window_size, self.output_3dmm_dim),
                                        device=self.device)
            reaction_emotion = torch.zeros((speaker_video.size(0), self.window_size, self.output_emotion_dim),
                                           device=self.device)

            for i in range(0, period):
                # mask = (~(num_windows < i)).view(-1, 1).repeat(1, self.window_size)
                # mask = mask.unsqueeze(-1).to(device=speaker_video)
                speaker_video_, speaker_audio_ = (speaker_video[:, :(i + 1) * self.window_size],
                                                  speaker_audio[:, :(i + 1) * self.window_size])
                encoded_feature = self.speaker_behaviour_encoder(speaker_video_, speaker_audio_)

                # modality fusion
                encoded_feature = self.fusion(
                    torch.cat((encoded_feature, reaction_3dmm, reaction_emotion), dim=-1))

                if i != 0:
                    past_reaction_3dmm, past_reaction_emotion = (reaction_3dmm[:, :i * self.window_size],
                                                                 reaction_emotion[:, :i * self.window_size])
                    current_reaction_3dmm, current_reaction_emotion = (reaction_3dmm[:, i * self.window_size:],
                                                                       reaction_emotion[:, i * self.window_size:])
                    listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature,
                                                                                          past_reaction_3dmm)

                    reaction_3dmm = torch.cat(
                        (past_reaction_3dmm, listener_3dmm_out, current_reaction_3dmm), dim=1)
                    reaction_emotion = torch.cat(
                        (past_reaction_emotion, listener_emotion_out, current_reaction_emotion), dim=1)

                else:
                    listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature)
                    reaction_3dmm = torch.cat((listener_3dmm_out, reaction_3dmm), dim=1)
                    reaction_emotion = torch.cat((listener_emotion_out, reaction_emotion), dim=1)

                if return_distribution:
                    distribution.append(dist)

            listener_3dmm_out, listener_emotion_out = reaction_3dmm[:, :frame_num], reaction_emotion[:, :frame_num]
            seq_mask = lengths_to_mask(lengths=motion_lengths,
                                       device=speaker_video.device,
                                       max_len=frame_num).unsqueeze(-1).float()
            listener_3dmm_out = listener_3dmm_out * seq_mask
            listener_emotion_out = listener_emotion_out * seq_mask

            listener_3dmm_out, listener_emotion_out = \
                list(listener_3dmm_out), list(listener_emotion_out)
            eeg_outputs = self._attach_eeg_outputs(
                listener_emotion_out,
                speaker_audio=speaker_audio,
                speaker_emotion=speaker_emotion,
                speaker_3dmm=speaker_3dmm,
                listener_eeg_input=listener_eeg_input,
                listener_eeg_mask=listener_eeg_mask,
                motion_lengths=motion_lengths,
            )
            if return_eeg_outputs:
                return listener_3dmm_out, listener_emotion_out, distribution, eeg_outputs
            return listener_3dmm_out, listener_emotion_out, distribution

        else:
            listener_3dmm_outs = []
            listener_emotion_outs = []
            for speaker_video_, speaker_audio_ in zip(speaker_video, speaker_audio):  # motion_lengths
                speaker_video_, speaker_audio_ = \
                    speaker_video_.unsqueeze(0).to(self.device), speaker_audio_.unsqueeze(0).to(self.device)
                encoded_feature = self.speaker_behaviour_encoder(speaker_video_, speaker_audio_)
                listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature)
                listener_3dmm_outs.append(listener_3dmm_out.squeeze(0))
                listener_emotion_outs.append(listener_emotion_out.squeeze(0))
                if return_distribution:
                    distribution.append(dist)

            eeg_outputs = self._attach_eeg_outputs(
                listener_emotion_outs,
                speaker_audio=speaker_audio,
                speaker_emotion=speaker_emotion,
                speaker_3dmm=speaker_3dmm,
                listener_eeg_input=listener_eeg_input,
                listener_eeg_mask=listener_eeg_mask,
                motion_lengths=kwargs.get('motion_lengths', None),
            )
            if return_eeg_outputs:
                return listener_3dmm_outs, listener_emotion_outs, distribution, eeg_outputs
            return listener_3dmm_outs, listener_emotion_outs, distribution

    def reset_window_size(self, window_size):
        self.window_size = window_size
        self.reaction_decoder.reset_window_size(window_size)

    def get_model_name(self):
        return 'TransformerVAE'
