import torch


def temporal_loss(y):
    diff = y[:, 1:, :] - y[:, :-1, :]
    return torch.mean(torch.norm(diff, dim=2, p=2) ** 2)


def l1_loss(prediction, target, reduction="min", **kwargs):
    assert prediction.shape == target.shape
    assert prediction.dim() == 3
    loss = torch.abs(prediction - target).mean(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "min":
        return loss.min(dim=-1)[0].mean()
    raise NotImplementedError(f"Unsupported reduction: {reduction}")


def mse_loss(prediction, target, reduction="mean", **kwargs):
    assert prediction.shape == target.shape
    assert prediction.dim() == 3
    loss = ((prediction - target) ** 2).mean(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "min":
        return loss.min(dim=-1)[0].mean()
    raise NotImplementedError(f"Unsupported reduction: {reduction}")


def masked_mse_loss(prediction, target, mask):
    mask = mask.to(dtype=prediction.dtype)
    loss = ((prediction - target) ** 2) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def DiffusionLoss(
        output_prior,
        output_decoder,
        losses_type=("MSELoss", "MSELoss"),
        losses_multipliers=(0.0, 1.0),
        losses_decoded=(False, True),
        k=1,
        temporal_loss_w=0.0,
        eeg_loss_weight=1.0,
        **kwargs):
    encoded_prediction = output_prior["encoded_prediction"]
    encoded_target = output_prior["encoded_target"]
    if encoded_prediction.dim() == 4:
        encoded_prediction = encoded_prediction.squeeze(-2)
    if encoded_target.dim() == 4:
        encoded_target = encoded_target.squeeze(-2)
    prediction_emotion = output_decoder["prediction_emotion"]
    target_emotion = output_decoder["target_emotion"]

    if prediction_emotion.dim() == 4:
        _, _, window_size, emotion_dim = prediction_emotion.shape
    else:
        _, window_size, emotion_dim = prediction_emotion.shape

    losses_dict = {"loss": prediction_emotion.new_tensor(0.0)}
    losses_dict["loss_eeg"] = prediction_emotion.new_tensor(0.0)
    losses_dict["eeg_valid_ratio"] = prediction_emotion.new_tensor(0.0)
    losses_dict["temporal_loss"] = temporal_loss(prediction_emotion.reshape(-1, window_size, emotion_dim))
    losses_dict["loss"] = losses_dict["loss"] + losses_dict["temporal_loss"] * temporal_loss_w

    prediction_emotion = prediction_emotion.reshape(-1, k, window_size * emotion_dim)
    target_emotion = target_emotion.reshape(-1, k, window_size * emotion_dim)

    loss_fns = {
        "MSELoss": mse_loss,
        "L1Loss": l1_loss,
    }
    for loss_name, weight, decoded in zip(losses_type, losses_multipliers, losses_decoded):
        key = "decoded" if decoded else "encoded"
        loss_fn = loss_fns[loss_name]
        if decoded:
            losses_dict[key] = loss_fn(prediction_emotion, target_emotion, k=k)
        else:
            losses_dict[key] = loss_fn(encoded_prediction, encoded_target, k=k)
        losses_dict["loss"] = losses_dict["loss"] + losses_dict[key] * weight

    if "prediction_eeg" in output_decoder and "target_eeg" in output_decoder:
        prediction_eeg = output_decoder["prediction_eeg"]
        target_eeg = output_decoder["target_eeg"]
        target_eeg_mask = output_decoder.get("target_eeg_mask", torch.ones_like(target_eeg))
        losses_dict["loss_eeg"] = masked_mse_loss(prediction_eeg, target_eeg, target_eeg_mask)
        losses_dict["eeg_valid_ratio"] = target_eeg_mask.float().mean()
        losses_dict["loss"] = losses_dict["loss"] + eeg_loss_weight * losses_dict["loss_eeg"]

    return losses_dict
