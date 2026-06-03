import time
import numpy as np
import multiprocessing as mp
import torch
from framework.metrics import *
from tslearn.metrics import dtw


def _func(target, pred):
    # target: (10, l, dim)
    # pred: (10, l, dim)
    num_preds = pred.shape[0]
    mean_mae_sum = 0
    for i in range(num_preds):
        mae_list = []
        for j in range(num_preds):
            mae = np.mean(np.abs(target[j][:, 15:].numpy() - pred[i][:, 15:].numpy()))
            mae_list.append(mae)
        mean_mae_sum += np.mean(mae_list)
    return mean_mae_sum / num_preds


def compute_MAE(preds, targets, p=4):
    MAE_list = []
    with mp.Pool(processes=p) as pool:
        MAE_list += pool.starmap(_func, zip(targets, preds))
    return np.mean(MAE_list)


def compute_metrics(speaker_inputs,
                    listener_predictions,
                    listener_targets,
                    threads=16):

    metrics = {
        'FRC': (compute_FRC, (listener_predictions, listener_targets), {'p': threads}),
        'FRD': (compute_FRD, (listener_predictions, listener_targets), {'p': threads}),
        'TLCC': (compute_TLCC, (listener_predictions, speaker_inputs),  {'p': threads}),
        'smse': (compute_s_mse, (listener_predictions,), {}),
        'FRVar': (compute_FRVar, (listener_predictions,), {}),
        # 'MAE': (compute_MAE, (listener_predictions, listener_targets), {'p': threads}),
    }

    results = {}
    for name, (func, args, kwargs) in metrics.items():
        t0 = time.perf_counter()
        value = func(*args, **kwargs)
        if hasattr(value, 'item'):
            value = value.item()
        elapsed = time.perf_counter() - t0

        results[name] = value
        results[f"{name}_time"] = elapsed
        print(f"{name:6s} = {value:.6f}  (time: {elapsed:.4f}s)")

    return results


def _to_3d_tensor(item):
    if isinstance(item, list):
        item = torch.stack(item, dim=0)
    item = torch.as_tensor(item).float()
    if item.dim() == 2:
        item = item.unsqueeze(0)
    return item


def _eeg_mask_for(mask, target_idx, length, dim):
    if mask is None:
        return torch.ones(length, dim)
    mask = _to_3d_tensor(mask)
    if mask.shape[0] == 1:
        mask_item = mask[0]
    else:
        mask_item = mask[min(target_idx, mask.shape[0] - 1)]
    return mask_item[:length, :dim].float()


def _masked_ccc_1d(target, prediction, mask):
    valid = mask > 0.5
    valid = valid & torch.isfinite(target) & torch.isfinite(prediction)
    if valid.sum().item() < 2:
        return None
    y_true = target[valid].detach().cpu().numpy()
    y_pred = prediction[valid].detach().cpu().numpy()
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true < 1e-8 or std_pred < 1e-8:
        return 0.0
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    return (2 * cor * std_true * std_pred) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def _masked_channel_mean_ccc(target, prediction, mask):
    scores = []
    for dim_idx in range(target.shape[-1]):
        score = _masked_ccc_1d(target[:, dim_idx], prediction[:, dim_idx], mask[:, dim_idx])
        if score is not None and np.isfinite(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else None


def _masked_channel_mean_dtw(target, prediction, mask):
    scores = []
    for dim_idx in range(target.shape[-1]):
        valid = mask[:, dim_idx] > 0.5
        valid = valid & torch.isfinite(target[:, dim_idx]) & torch.isfinite(prediction[:, dim_idx])
        if valid.sum().item() < 1:
            continue
        target_ch = target[valid, dim_idx].detach().cpu().numpy().astype(np.float32)
        pred_ch = prediction[valid, dim_idx].detach().cpu().numpy().astype(np.float32)
        scores.append(dtw(pred_ch, target_ch))
    return float(np.mean(scores)) if scores else None


def _compute_eeg_frc(preds, targets, masks):
    sample_scores = []
    for sample_idx, (prediction, target) in enumerate(zip(preds, targets)):
        prediction = _to_3d_tensor(prediction)
        target = _to_3d_tensor(target)
        mask = masks[sample_idx] if masks is not None else None
        length = min(prediction.shape[1], target.shape[1])
        dim = min(prediction.shape[2], target.shape[2])
        prediction = prediction[:, :length, :dim]
        target = target[:, :length, :dim]

        pred_scores = []
        for pred_idx in range(prediction.shape[0]):
            target_scores = []
            for target_idx in range(target.shape[0]):
                target_mask = _eeg_mask_for(mask, target_idx, length, dim)
                score = _masked_channel_mean_ccc(target[target_idx], prediction[pred_idx], target_mask)
                if score is not None:
                    target_scores.append(score)
            if target_scores:
                pred_scores.append(max(target_scores))
        if pred_scores:
            sample_scores.append(float(np.sum(pred_scores)))
    return float(np.mean(sample_scores)) if sample_scores else float("nan")


def _compute_eeg_frd(preds, targets, masks):
    sample_scores = []
    for sample_idx, (prediction, target) in enumerate(zip(preds, targets)):
        prediction = _to_3d_tensor(prediction)
        target = _to_3d_tensor(target)
        mask = masks[sample_idx] if masks is not None else None
        length = min(prediction.shape[1], target.shape[1])
        dim = min(prediction.shape[2], target.shape[2])
        prediction = prediction[:, :length, :dim]
        target = target[:, :length, :dim]

        pred_scores = []
        for pred_idx in range(prediction.shape[0]):
            target_scores = []
            for target_idx in range(target.shape[0]):
                target_mask = _eeg_mask_for(mask, target_idx, length, dim)
                score = _masked_channel_mean_dtw(target[target_idx], prediction[pred_idx], target_mask)
                if score is not None:
                    target_scores.append(score)
            if target_scores:
                pred_scores.append(min(target_scores))
        if pred_scores:
            sample_scores.append(float(np.sum(pred_scores)))
    return float(np.mean(sample_scores)) if sample_scores else float("nan")


def _compute_eeg_tlcc(preds, targets, masks, seconds=2, fps=1):
    sample_offsets = []
    max_lag = int(seconds * fps)
    for sample_idx, (prediction, target) in enumerate(zip(preds, targets)):
        prediction = _to_3d_tensor(prediction)
        target = _to_3d_tensor(target)
        mask = masks[sample_idx] if masks is not None else None
        length = min(prediction.shape[1], target.shape[1])
        dim = min(prediction.shape[2], target.shape[2])
        prediction = prediction[:, :length, :dim]
        target = target[0, :length, :dim]
        target_mask = _eeg_mask_for(mask, 0, length, dim)

        pred_offsets = []
        for pred_idx in range(prediction.shape[0]):
            channel_offsets = []
            for dim_idx in range(dim):
                lag_scores = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag > 0:
                        pred_ch = prediction[pred_idx, lag:, dim_idx]
                        target_ch = target[:-lag, dim_idx]
                        mask_ch = target_mask[:-lag, dim_idx]
                    elif lag < 0:
                        pred_ch = prediction[pred_idx, :lag, dim_idx]
                        target_ch = target[-lag:, dim_idx]
                        mask_ch = target_mask[-lag:, dim_idx]
                    else:
                        pred_ch = prediction[pred_idx, :, dim_idx]
                        target_ch = target[:, dim_idx]
                        mask_ch = target_mask[:, dim_idx]
                    ccc = _masked_ccc_1d(target_ch, pred_ch, mask_ch)
                    if ccc is not None:
                        lag_scores.append((ccc, lag))
                if lag_scores:
                    best_lag = max(lag_scores, key=lambda item: item[0])[1]
                    channel_offsets.append(abs(best_lag))
            if channel_offsets:
                pred_offsets.append(float(np.mean(channel_offsets)))
        if pred_offsets:
            sample_offsets.append(float(np.mean(pred_offsets)))
    return float(np.mean(sample_offsets)) if sample_offsets else float("nan")


def _compute_eeg_smse(preds):
    distances = []
    for prediction in preds:
        prediction = _to_3d_tensor(prediction)
        if prediction.shape[0] < 2:
            distances.append(prediction.new_tensor(0.0))
            continue
        flattened = prediction.reshape(prediction.shape[0], -1)
        dist = torch.pow(torch.cdist(flattened, flattened), 2)
        dist = torch.sum(dist) / (flattened.shape[0] * (flattened.shape[0] - 1) * flattened.shape[1])
        distances.append(dist)
    return torch.stack(distances).mean().item() if distances else float("nan")


def _compute_eeg_frvar(preds, masks=None):
    values = []
    for sample_idx, prediction in enumerate(preds):
        prediction = _to_3d_tensor(prediction)
        if masks is None:
            values.append(torch.var(prediction, dim=1, unbiased=False).mean())
            continue
        mask = _to_3d_tensor(masks[sample_idx])[0]
        length = min(prediction.shape[1], mask.shape[0])
        dim = min(prediction.shape[2], mask.shape[1])
        valid = mask[:length, :dim] > 0.5
        if valid.sum().item() == 0:
            continue
        var = torch.var(prediction[:, :length, :dim], dim=1, unbiased=False)
        channel_valid = valid.any(dim=0)
        if channel_valid.any():
            values.append(var[:, channel_valid].mean())
    return torch.stack(values).mean().item() if values else float("nan")


def compute_eeg_metrics(listener_predictions,
                        listener_targets,
                        masks=None,
                        threads=16):
    metrics = {
        "EEG_FRC": lambda: _compute_eeg_frc(listener_predictions, listener_targets, masks),
        "EEG_FRD": lambda: _compute_eeg_frd(listener_predictions, listener_targets, masks),
        "EEG_TLCC": lambda: _compute_eeg_tlcc(listener_predictions, listener_targets, masks),
        "EEG_smse": lambda: _compute_eeg_smse(listener_predictions),
        "EEG_FRVar": lambda: _compute_eeg_frvar(listener_predictions, masks),
    }
    results = {}
    for name, func in metrics.items():
        t0 = time.perf_counter()
        value = func()
        elapsed = time.perf_counter() - t0
        if hasattr(value, "item"):
            value = value.item()
        results[name] = value
        results[f"{name}_time"] = elapsed
        print(f"{name:10s} = {value:.6f}  (time: {elapsed:.4f}s)")
    return results
