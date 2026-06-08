import argparse
import random
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from framework.metrics import compute_FRC, compute_FRD, compute_FRVar, compute_TLCC, compute_s_mse
from framework.metrics.metric import baseline_mime, baseline_random
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_MAE


BASELINE_NAMES = ("GT_identical", "B_Random", "B_Mime", "B_MeanFr")


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _is_sample_file(path):
    return (
        path.suffix.lower() == ".npy"
        and not path.name.startswith(".")
        and not path.name.startswith("._")
    )


def _load_emotion(path):
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise ValueError(
            f"Cannot load numeric facial-attribute file: {path}. "
            "It may be an AppleDouble/archive metadata file."
        ) from exc

    tensor = torch.from_numpy(np.asarray(array)).float()
    if tensor.dim() > 2:
        tensor = tensor.squeeze()
    if tensor.dim() == 1 and tensor.numel() == 25:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.shape[-1] != 25:
        raise ValueError(f"Expected facial attributes with shape [T, 25], got {tuple(tensor.shape)} at {path}")
    return tensor


def _iter_role_files(facial_dir, role):
    role_dir = facial_dir / role
    if not role_dir.is_dir():
        return
    for session_dir in sorted(path for path in role_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        for path in sorted(session_dir.iterdir()):
            if _is_sample_file(path):
                yield Path(role) / session_dir.name / path.stem


def _build_gt_index(facial_dir):
    index = {}
    for role in ("speaker", "listener"):
        for rel_path in _iter_role_files(facial_dir, role):
            session_key = Path(rel_path.parts[0]) / rel_path.parts[1]
            index.setdefault(session_key, []).append(rel_path)
    return index


def _rel_to_path(facial_dir, rel_path):
    return facial_dir / rel_path.with_suffix(".npy")


def _build_samples(facial_dir, num_preds, bidirectional, rng):
    gt_index = _build_gt_index(facial_dir)
    roles = ("speaker", "listener") if bidirectional else ("speaker",)
    samples = []
    skipped = 0

    for role in roles:
        target_role = "listener" if role == "speaker" else "speaker"
        for input_rel in _iter_role_files(facial_dir, role):
            session = input_rel.parts[1]
            target_rel = Path(target_role) / session / input_rel.name
            if not _rel_to_path(facial_dir, target_rel).is_file():
                warnings.warn(f"Skip missing paired target: {target_rel}")
                skipped += 1
                continue

            target_session = Path(target_role) / session
            candidates = list(gt_index.get(target_session, []))
            extra_candidates = [path for path in candidates if path != target_rel]
            extra_count = max(num_preds - 1, 0)
            if extra_count == 0:
                gt_rels = [target_rel]
            elif len(extra_candidates) >= extra_count:
                gt_rels = [target_rel] + rng.sample(extra_candidates, extra_count)
            elif extra_candidates:
                gt_rels = [target_rel] + rng.choices(extra_candidates, k=extra_count)
            else:
                gt_rels = [target_rel] + [target_rel] * extra_count

            samples.append(
                {
                    "input": input_rel,
                    "target": target_rel,
                    "gts": gt_rels,
                }
            )

    return samples, skipped


def _load_samples(facial_dir, samples, max_samples=None):
    speaker_inputs = []
    raw_listener_targets = []
    sample_ids = []
    skipped = 0

    selected = samples[:max_samples] if max_samples is not None else samples
    for sample in selected:
        try:
            speaker_input = _load_emotion(_rel_to_path(facial_dir, sample["input"]))
            targets = [_load_emotion(_rel_to_path(facial_dir, rel_path)) for rel_path in sample["gts"]]
        except (OSError, ValueError) as exc:
            warnings.warn(f"Skip invalid sample {sample['input']} -> {sample['target']}: {exc}")
            skipped += 1
            continue

        speaker_inputs.append(speaker_input)
        raw_listener_targets.append(targets)
        sample_ids.append(
            {
                "input": str(sample["input"]),
                "target": str(sample["target"]),
                "gts": [str(path) for path in sample["gts"]],
            }
        )

    return speaker_inputs, raw_listener_targets, sample_ids, skipped


def _compute_train_mean_fr(facial_dir):
    listener_dir = facial_dir / "listener"
    if not listener_dir.is_dir():
        raise FileNotFoundError(f"Missing training listener facial-attributes directory: {listener_dir}")

    sum_vector = torch.zeros(25, dtype=torch.float64)
    frame_count = 0
    file_count = 0
    for rel_path in _iter_role_files(facial_dir, "listener"):
        path = _rel_to_path(facial_dir, rel_path)
        emotion = _load_emotion(path).double()
        sum_vector += emotion.sum(dim=0)
        frame_count += emotion.shape[0]
        file_count += 1

    if frame_count == 0:
        raise RuntimeError(f"No listener facial-attribute frames found under {listener_dir}")

    print(f"Training listener mean uses {file_count} files and {frame_count} frames.")
    return (sum_vector / frame_count).float()


def _build_target_alignment_predictions(speaker_inputs, num_preds):
    return [
        speaker_input.new_zeros((num_preds, speaker_input.shape[0], speaker_input.shape[1]))
        for speaker_input in speaker_inputs
    ]


def _make_gt_identical(listener_targets):
    return [
        target.clone()
        for target in tqdm(listener_targets, desc="Building GT_identical", leave=False)
    ]


def _make_random(listener_targets):
    return [
        baseline_random(target)
        for target in tqdm(listener_targets, desc="Building B_Random", leave=False)
    ]


def _make_mime(speaker_inputs, num_preds):
    predictions = []
    for speaker_input in tqdm(speaker_inputs, desc="Building B_Mime", leave=False):
        if num_preds == 10:
            predictions.append(baseline_mime(speaker_input))
        else:
            predictions.append(speaker_input.unsqueeze(0).expand(num_preds, -1, -1).clone())
    return predictions


def _make_meanfr(speaker_inputs, train_mean_fr, num_preds):
    mean_fr = train_mean_fr.view(1, 1, -1)
    return [
        mean_fr.expand(num_preds, speaker_input.shape[0], -1).clone()
        for speaker_input in tqdm(speaker_inputs, desc="Building B_MeanFr", leave=False)
    ]


def _predictions_equal_targets(listener_predictions, listener_targets, atol=1e-6):
    if len(listener_predictions) != len(listener_targets):
        return False

    for prediction, target in zip(listener_predictions, listener_targets):
        if prediction.shape != target.shape:
            return False
        if not torch.allclose(prediction, target, atol=atol, rtol=0.0):
            return False
    return True


def _identity_frc(listener_predictions):
    num_preds = [prediction.shape[0] for prediction in listener_predictions]
    return float(np.mean(num_preds))


def compute_reaction_metrics(
        speaker_inputs,
        listener_predictions,
        listener_targets,
        threads=16,
        desc="Metrics",
        force_identity_frc=False):
    metrics = OrderedDict(
        [
            ("FRC", (compute_FRC, (listener_predictions, listener_targets), {"p": threads})),
            ("FRD", (compute_FRD, (listener_predictions, listener_targets), {"p": threads})),
            ("TLCC", (compute_TLCC, (listener_predictions, speaker_inputs), {"p": threads})),
            ("smse", (compute_s_mse, (listener_predictions,), {})),
            ("FRVar", (compute_FRVar, (listener_predictions,), {})),
            ("MAE", (compute_MAE, (listener_predictions, listener_targets), {"p": threads})),
        ]
    )

    results = {}
    for name, (func, args, kwargs) in tqdm(metrics.items(), desc=desc):
        t0 = time.perf_counter()
        if name == "FRC" and force_identity_frc:
            if _predictions_equal_targets(listener_predictions, listener_targets):
                # The project FRC uses CCC. For exactly identical constant channels
                # (common in binary AU streams), CCC is undefined and the metric code
                # currently converts it to 0. GT_identical is an oracle baseline, so
                # after verifying pred == target we report the expected perfect count.
                value = _identity_frc(listener_predictions)
            else:
                warnings.warn(
                    "force_identity_frc=True but predictions differ from targets; "
                    "falling back to the standard FRC implementation."
                )
                value = func(*args, **kwargs)
        else:
            value = func(*args, **kwargs)
        if hasattr(value, "item"):
            value = value.item()
        elapsed = time.perf_counter() - t0
        results[name] = float(value)
        results[f"{name}_time"] = elapsed
        print(f"{name:6s} = {float(value):.6f}  (time: {elapsed:.4f}s)")
    return results


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run no-training REACT facial-reaction baselines."
    )
    parser.add_argument("--data_dir", required=True, help="REACT2025 data root containing train/val/test splits.")
    parser.add_argument("--split", default="test", help="Evaluation split name.")
    parser.add_argument("--train_split", default="train", help="Split used to compute B_MeanFr.")
    parser.add_argument("--num_preds", type=int, default=10, help="Number of prediction samples per test item.")
    parser.add_argument("--threads", type=int, default=16, help="Metric multiprocessing worker count.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for GT sampling and B_Random.")
    parser.add_argument("--bidirectional", action="store_true", help="Evaluate both speaker->listener and listener->speaker.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument("--output_path", default="baseline_reaction_results.pt", help="Output .pt path.")
    parser.add_argument("--post_config_name", default="configs/shared/model/emotion_autoencoder.yaml")
    parser.add_argument("--post_ckpt_dir", default=None, help="Post-processor checkpoint directory.")
    parser.add_argument("--post_clip_length", type=int, default=1000)
    parser.add_argument("--device", default=None, help="Post-processor device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.num_preds <= 1:
        raise ValueError("--num_preds must be greater than 1 because smse requires multiple predictions.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = _repo_root()
    data_dir = _resolve_path(args.data_dir)
    eval_facial_dir = data_dir / args.split / "facial-attributes"
    train_facial_dir = data_dir / args.train_split / "facial-attributes"
    if not eval_facial_dir.is_dir():
        raise FileNotFoundError(f"Missing evaluation facial-attributes directory: {eval_facial_dir}")
    if not train_facial_dir.is_dir():
        raise FileNotFoundError(f"Missing training facial-attributes directory: {train_facial_dir}")

    post_ckpt_dir = _resolve_path(args.post_ckpt_dir) if args.post_ckpt_dir else repo_root / "pretrained_models" / "post_processor"
    if not (post_ckpt_dir / "checkpoint.pth").is_file():
        raise FileNotFoundError(
            "Missing post-processor checkpoint. "
            f"Expected: {post_ckpt_dir / 'checkpoint.pth'}"
        )

    rng = random.Random(args.seed)
    samples, pair_skipped = _build_samples(
        eval_facial_dir,
        num_preds=args.num_preds,
        bidirectional=args.bidirectional,
        rng=rng,
    )
    if not samples:
        raise RuntimeError(f"No valid paired samples found under {eval_facial_dir}")

    speaker_inputs, raw_listener_targets, sample_ids, load_skipped = _load_samples(
        eval_facial_dir,
        samples,
        max_samples=args.max_samples,
    )
    if not speaker_inputs:
        raise RuntimeError("No samples could be loaded after filtering invalid files.")

    print(
        f"Loaded {len(speaker_inputs)} samples from split={args.split}; "
        f"pair_skipped={pair_skipped}; load_skipped={load_skipped}; "
        f"bidirectional={args.bidirectional}."
    )

    train_mean_fr = _compute_train_mean_fr(train_facial_dir)

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    post_processor = Processor(
        config_name=args.post_config_name,
        ckpt_dir=str(post_ckpt_dir),
        cfg_dir=str(repo_root),
        clip_len_test=args.post_clip_length,
        device=device,
        num_preds=args.num_preds,
    )
    target_alignment_predictions = _build_target_alignment_predictions(speaker_inputs, args.num_preds)
    listener_targets = post_processor.forward(
        prediction_list=target_alignment_predictions,
        target_list=raw_listener_targets,
    )

    baseline_predictions = OrderedDict(
        [
            ("GT_identical", _make_gt_identical(listener_targets)),
            ("B_Random", _make_random(listener_targets)),
            ("B_Mime", _make_mime(speaker_inputs, args.num_preds)),
            ("B_MeanFr", _make_meanfr(speaker_inputs, train_mean_fr, args.num_preds)),
        ]
    )

    metrics = OrderedDict()
    for name, predictions in baseline_predictions.items():
        print(f"\n=== {name} ===")
        metrics[name] = compute_reaction_metrics(
            speaker_inputs=speaker_inputs,
            listener_predictions=predictions,
            listener_targets=listener_targets,
            threads=args.threads,
            desc=f"Evaluating {name}",
            force_identity_frc=(name == "GT_identical"),
        )

    output = {
        "GT": listener_targets,
        "INPUT": speaker_inputs,
        "PRED": baseline_predictions,
        "metrics": metrics,
        "train_mean_fr": train_mean_fr,
        "sample_ids": sample_ids,
        "config": {
            "data_dir": str(data_dir),
            "split": args.split,
            "train_split": args.train_split,
            "num_preds": args.num_preds,
            "bidirectional": args.bidirectional,
            "seed": args.seed,
            "threads": args.threads,
            "max_samples": args.max_samples,
        },
    }

    output_path = _resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"\nSaved baseline results to {output_path}")


if __name__ == "__main__":
    main()
