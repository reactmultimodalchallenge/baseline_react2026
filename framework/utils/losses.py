from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class VAELoss(nn.Module):
    def __init__(self, kl_p: float = 0.0002,
                 w_emo: float = None,
                 w_exp: float = None,
                 w_rot: float = None,
                 w_tran: float = None,
                 eeg_loss_weight: float = 1.0,
                 **kwargs):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.kl_loss = KLLoss()
        self.kl_p = kl_p
        self.eeg_loss_weight = eeg_loss_weight

        if w_emo is None:
            w_emo = 1.0
        if w_exp is None:
            w_exp = 1.0
        if w_rot is None:
            w_rot = 10.0
        if w_tran is None:
            w_tran = 10.0
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran

    @staticmethod
    def masked_mse(prediction, target, mask):
        mask = mask.to(dtype=prediction.dtype)
        loss = ((prediction - target) ** 2) * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    def forward(self, gt_emotions, gt_3dmms, pred_emotions, pred_3dmms, distribution,
                prediction_eeg=None, target_eeg=None, target_eeg_mask=None):
        """ List
        gt_emotion; gt_3dmm; pred_emotion; pred_3dmm
        """
        bsz = len(gt_emotions)
        rec_emotion_loss = 0
        rec_param_loss = 0
        for gt_emotion, gt_3dmm, pred_emotion, pred_3dmm in zip(
                gt_emotions, gt_3dmms, pred_emotions, pred_3dmms):
            gt_emotion = gt_emotion.to(pred_emotion.get_device())
            gt_3dmm = gt_3dmm.to(pred_3dmm.get_device())

            exp_part = self.w_exp * self.mse(pred_3dmm[:, :52], gt_3dmm[:, :52])  # expression
            rot_part = self.w_rot * self.mse(pred_3dmm[:, 52:55], gt_3dmm[:, 52:55])  # rotation
            tran_part = self.w_tran * self.mse(pred_3dmm[:, 55:], gt_3dmm[:, 55:])  # translation
            rec_param_loss = rec_param_loss + (exp_part + rot_part + tran_part)
            rec_emotion_loss = rec_emotion_loss + self.w_emo * self.mse(pred_emotion, gt_emotion)

        rec_emotion_loss = rec_emotion_loss / bsz
        rec_param_loss = rec_param_loss / bsz
        rec_loss = rec_emotion_loss + rec_param_loss

        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_emotion.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_emotion.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss
        loss_eeg = loss.new_tensor(0.0)
        eeg_valid_ratio = loss.new_tensor(0.0)
        if prediction_eeg is not None and target_eeg is not None:
            target_eeg = target_eeg.to(prediction_eeg.device).float()
            target_eeg_mask = target_eeg_mask.to(prediction_eeg.device).float() \
                if target_eeg_mask is not None else torch.ones_like(target_eeg)
            loss_eeg = self.masked_mse(prediction_eeg, target_eeg, target_eeg_mask)
            eeg_valid_ratio = target_eeg_mask.float().mean()
            loss = loss + self.eeg_loss_weight * loss_eeg
        return loss, rec_loss, rec_emotion_loss, rec_param_loss, kld_loss, loss_eeg, eeg_valid_ratio

    def __repr__(self):
        return "VAELoss()"


def div_loss(Y_1_list, Y_2_list):
    loss = 0.0
    B = len(Y_1_list)
    for y1, y2 in zip(Y_1_list, Y_2_list):
        y1_flat = y1.view(1, -1)
        y2_flat = y2.view(1, -1)
        Y = torch.cat([y1_flat, y2_flat], dim=0)
        dist2 = F.pdist(Y, 2) ** 2
        loss += (-dist2 / 100).exp().mean()
    loss = loss / B
    return loss


def div_loss_v2(Y_1, Y_2):
    loss = 0.0
    b,t,c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b,1,-1), Y_2.view(b,1,-1)], dim = 1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist /  100).exp().mean()
    loss /= b
    return loss


def TemporalLoss(Y):
    diff = Y[:, 1:, :] - Y[:, :-1, :]
    t_loss = torch.mean(torch.norm(diff, dim=2, p=2) ** 2)
    return t_loss


def L1Loss(prediction, target, reduction="min", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of L1 loss
    loss = (torch.abs(prediction - target)).mean(dim=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(dim=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELoss(prediction, target, reduction="mean", **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    loss = ((prediction - target) ** 2).mean(dim=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(dim=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELossApt(prediction, target, reduction="mean",
               w_au=1.0, w_va=5.0, w_em=2.0, **kwargs):
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"
    loss_au = F.mse_loss(prediction[:, :, :15], target[:, :, :15], reduction=reduction)
    loss_va = F.mse_loss(prediction[:, :, 15:17], target[:, :, 15:17], reduction=reduction)
    loss_em = F.mse_loss(prediction[:, :, 17:], target[:, :, 17:], reduction=reduction)
    loss = loss_au * w_au + loss_va * w_va + loss_em * w_em

    losses_dict = {"loss": loss, "loss_au": loss_au, "loss_va": loss_va, "loss_em": loss_em}
    return losses_dict


class DiffusionLoss:
    def __init__(self,
                 losses_type='MSELoss',
                 n_preds=10,
                 prior_loss_weight=1.0,
                 eeg_loss_weight=1.0,
                 w_au=1.0,  # action unit
                 w_va=5.0,  # valence and arousal
                 w_em=2.0,  # emotion
                 **kwargs):
        self.loss_type = losses_type
        self.n_preds = n_preds
        self.prior_loss_weight = prior_loss_weight
        self.eeg_loss_weight = eeg_loss_weight
        self.w_au = w_au
        self.w_va = w_va
        self.w_em = w_em

    @staticmethod
    def masked_mse(prediction, target, mask):
        mask = mask.to(dtype=prediction.dtype)
        loss = ((prediction - target) ** 2) * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    def __call__(self, output_decoder):
        output_prior = None
        if "output_decoder" in output_decoder:
            output_prior = output_decoder.get("output_prior")
            output_decoder = output_decoder["output_decoder"]

        prediction_emotion = output_decoder["prediction_emotion"]
        target_emotion = output_decoder["target_emotion"]

        _, _, window_size, emotion_dim = prediction_emotion.shape
        prediction_emotion = prediction_emotion.reshape(-1, self.n_preds, window_size * emotion_dim)
        target_emotion = target_emotion.reshape(-1, self.n_preds, window_size * emotion_dim)
        losses_dict = eval(self.loss_type)(
            prediction_emotion, target_emotion, k=self.n_preds, w_au=self.w_au, w_va=self.w_va, w_em=self.w_em)

        losses_dict["loss_eeg"] = losses_dict["loss"].new_tensor(0.0)
        losses_dict["eeg_valid_ratio"] = losses_dict["loss"].new_tensor(0.0)
        if "prediction_eeg" in output_decoder and "target_eeg" in output_decoder:
            prediction_eeg = output_decoder["prediction_eeg"]
            target_eeg = output_decoder["target_eeg"]
            target_eeg_mask = output_decoder.get("target_eeg_mask", torch.ones_like(target_eeg))
            loss_eeg = self.masked_mse(prediction_eeg, target_eeg, target_eeg_mask)
            losses_dict["loss_eeg"] = loss_eeg
            losses_dict["eeg_valid_ratio"] = target_eeg_mask.float().mean()
            losses_dict["loss"] = losses_dict["loss"] + self.eeg_loss_weight * loss_eeg

        if output_prior is not None:
            encoded_prediction = output_prior["encoded_prediction"]
            encoded_target = output_prior["encoded_target"]
            if encoded_prediction.dim() == 4:
                encoded_prediction = encoded_prediction.squeeze(-2)
            if encoded_target.dim() == 4:
                encoded_target = encoded_target.squeeze(-2)
            loss_prior = MSELoss(encoded_prediction, encoded_target, reduction="mean")
            losses_dict["loss_prior"] = loss_prior
            losses_dict["loss"] = losses_dict["loss"] + self.prior_loss_weight * loss_prior
        else:
            losses_dict["loss_prior"] = losses_dict["loss"].new_tensor(0.0)

        return losses_dict

class EmotionVAELoss:
    def __init__(self, w_au, w_va, w_em, w_kld, **kwargs):
        self.w_au = w_au
        self.w_va = w_va
        self.w_em = w_em
        self.w_kld = w_kld

        self.au_criterion = nn.BCEWithLogitsLoss(reduction="none")  # "mean"
        self.va_criterion = nn.MSELoss(reduction="mean")
        self.em_criterion = nn.KLDivLoss(reduction="none")  # batchmean
        self.kld_criterion = KLLoss()

    def kld_loss(self, mu, logvar, reduction='batchmean'):
        kld_element = 1 + logvar - mu.pow(2) - logvar.exp()
        if reduction == 'sum':
            # sum over B, L, D
            return -0.5 * torch.sum(kld_element)
        elif reduction == 'batchmean':
            # sum over L,D, then mean over batch
            kld = -0.5 * torch.sum(kld_element, dim=(1, 2))  # [B]
            return torch.mean(kld)  # scalar
        elif reduction == 'mean':
            # mean over all elements
            return -0.5 * torch.mean(kld_element)
        else:
            raise ValueError("Unknown reduction")

    def __call__(self, predictions, targets, distribution, mask=None):
        au_logits, va_logits, emotion_logits = predictions
        au_targets, va_targets, emotion_targets = \
            targets[:, :, :15], targets[:, :, 15:17], targets[:, :, 17:]

        au_loss = self.au_criterion(au_logits, au_targets)  # AUs
        va_loss = self.va_criterion(va_logits, va_targets)  # valence and arousal
        emotion_logits = rearrange(F.log_softmax(emotion_logits, dim=-1), "b l d -> (b l) d")
        emotion_targets = rearrange(emotion_targets, "b l d -> (b l) d")
        em_loss = self.em_criterion(emotion_logits, emotion_targets)

        if mask is not None:
            au_mask = mask  # [B, L, 1]
            au_loss = (au_loss * au_mask).sum() / (au_mask.sum() * au_logits.shape[-1])
            em_mask = rearrange(mask, "b l d -> (b l) d")  # [BxL, 1]
            em_loss = (em_loss * em_mask).sum() / em_mask.sum()

        mu_ref = torch.zeros_like(distribution.loc).to(au_logits)
        scale_ref = torch.ones_like(distribution.scale).to(au_logits)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kld_loss = self.kld_criterion(distribution, distribution_ref)

        loss = self.w_kld * kld_loss + \
               self.w_au * au_loss + \
               self.w_va * va_loss + \
               self.w_em * em_loss

        return loss, kld_loss, au_loss, va_loss, em_loss
