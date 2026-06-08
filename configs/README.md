# Config Organization

The task-facing Hydra configs are organized by challenge setting. Use these files with `python main.py --config-name <section>/<method> ...`.

| Section | Entry config | Training / evaluation entry | Model config files | Model code | Checkpoints |
| --- | --- | --- | --- | --- | --- |
| Generic online | `generic_online/motion_diffusion.yaml` | `main.py` with `stage=fit` or `stage=test` | `generic_online/model/motion_diffusion.yaml`, `generic_online/model/losses/motion_diffusion.yaml` | `trainer/motion_diffusion.py`, `framework/motion_diffusion/` | `save/motion_diffusion/react_2025/online/checkpoints/<run_id>/` |
| Generic online | `generic_online/motion_transvae.yaml` | `main.py` with `stage=fit` or `stage=test` | `generic_online/model/motion_transvae.yaml`, `generic_online/model/losses/motion_transvae.yaml` | `trainer/motion_transvae.py`, `framework/motion_transvae/` | `save/motion_transvae/react_2025/online/checkpoints/<run_id>/` |
| Generic offline | `generic_offline/motion_diffusion.yaml` | `main.py` with `stage=fit` or `stage=test` | `generic_offline/model/motion_diffusion.yaml`, `generic_offline/model/losses/motion_diffusion.yaml` | `trainer/motion_diffusion.py`, `framework/motion_diffusion/` | `save/motion_diffusion/react_2025/offline/checkpoints/<run_id>/` |
| Generic offline | `generic_offline/motion_transvae.yaml` | `main.py` with `stage=fit` or `stage=test` | `generic_offline/model/motion_transvae.yaml`, `generic_offline/model/losses/motion_transvae.yaml` | `trainer/motion_transvae.py`, `framework/motion_transvae/` | `save/motion_transvae/react_2025/offline/checkpoints/<run_id>/` |
| Generic offline | RegNN | `regnn/train.py`; add `--test` for evaluation | RegNN uses command-line arguments instead of Hydra YAML | `regnn/models/`, `regnn/trainers.py` | `regnn/<logs-dir>/mhp-*-seed<seed>.pth` or `regnn/<logs-dir>/mhp-eeg-head-*-seed<seed>.pth` |
| Personalized online | `personalized_online/perfrdiff_rewrite_weight.yaml` | `main.py` with `stage=fit` or `stage=test` | `personalized_online/model/motion_diffusion.yaml`, `personalized_online/model/losses/perfrdiff_rewrite_weight.yaml` | `trainer/perfrdiff_rewrite_weight.py`, `framework/perfrdiff_rewrite_weight/` | `save/perfrdiff_rewrite_weight/react_2025/online/checkpoints/<run_id>/` |
| Personalized offline | `personalized_offline/perfrdiff_rewrite_weight.yaml` | `main.py` with `stage=fit` or `stage=test` | `personalized_offline/model/motion_diffusion.yaml`, `personalized_offline/model/losses/perfrdiff_rewrite_weight.yaml` | `trainer/perfrdiff_rewrite_weight.py`, `framework/perfrdiff_rewrite_weight/` | `save/perfrdiff_rewrite_weight/react_2025/offline/checkpoints/<run_id>/` |

## Directory Notes

- `data/`, `trainer/`, `model/`, and `model/losses/` under each section contain the task-specific YAML used by that section.
- `shared/` contains global support configs: `data/`, `logger/`, `model/`, `trainer/`, and `path.yaml`.
- `configs/shared/path.yaml` exposes resolver-based paths such as the code root and current working directory, so it is kept with the other shared configs.
- `configs/shared/model/emotion_autoencoder.yaml` is intentionally kept as a shared config because the post-processor loads it by this path.

## Running

- Hydra training: `python main.py --config-name <section>/<method> stage=fit data_dir=<dataset-root>`
- Hydra evaluation: `python main.py --config-name <section>/<method> stage=test data_dir=<dataset-root> resume_id=<run-id>`
- RegNN training: run `python train.py ...` from the `regnn/` directory.
- RegNN evaluation: run `python train.py --test ...` from the `regnn/` directory.
