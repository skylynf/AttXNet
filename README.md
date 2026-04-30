# AttXNet

> The content of this repository corresponds to a paper currently under review at *IEEE Access* and is provided solely for review purposes. 


Surface crack image binary classification for bridge (infrastructure) scenes: **backbones (ResNet18 / MobileNetV3 / EfficientNet-B0)** + **optional attention (CBAM / CA)**, with **cross-entropy / weighted CE / Focal Loss**, **robust augmentation**, and **stratified splits / K-fold**, targeting SDNET2018-style `{D,P,W}` folder layouts.

This repo keeps actual experiment artifacts under `runs/` (metric JSON, configs, TensorBoard logs, etc.; `***.pth` weights may be omitted from the repo for size**—if missing, retrain with the same `config.json` to reproduce the pipeline). `experiments/` provides Bash batch entry points aligned with the paper / revision for one-shot reproduction of the experiment matrix.

**Open a terminal at the repository root (this README’s directory) before running commands**, so paths to the `attxnet` package, `train.py`, and `--output_dir` stay consistent.

---

## Repository layout


| Path                | Description                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `attxnet/`          | Importable core library: `dataset`, `models`, `losses`                                                                       |
| `train.py`          | Train / eval entry: writes `results.json`, TensorBoard, `best_model.pth` / `last_model.pth`                                  |
| `scripts/`          | Complexity stats, Grad-CAM, result summaries, `rev_*` aggregation, publication figures                                       |
| `experiments/`      | **Batch experiment** Bash scripts (see dedicated section); run under **Git Bash / Linux / WSL**                              |
| `runs/`             | **Historical experiment root** (below): typically `runs_v3/` (main + reviewer ablations), `runs_cv5/` (5-fold pairing), etc. |
| `outputs/revision/` | Reviewer supplementary assets after `scripts/export_revision_paper_assets.py`                                                |


Scripts often set `--output_dir` to `./runs_v3` or `./runs_cv5` (under the repo root); this repo also bundles prior outputs under `**runs/runs_v3` and `runs/runs_cv5`**. **When running analysis scripts, point `--runs_dir` at the level that actually contains `results.json`** (e.g. `./runs/runs_v3` or `./runs_v3`).

---

## `runs/`: artifacts and weights

`runs/` groups outputs from each training run for tables and figures. A typical layout (from what is committed here):

```text
runs/
  runs_v3/                    # single runs / ablations / multi-backbone / revision extras
    exp1_resnet18_baseline/
    exp2_ablation_full_cbam/
    rev_baseline_ce/
    rev_focal_gamma_2/
    ...
    figures/                  # tables, curves, publication plots from scripts
  runs_cv5/                   # stratified 5-fold CV (paired comparisons)
    cv5_resnet18_baseline/
      fold_0/
      fold_1/
      ...
    cv5_resnet18_full_cbam/
      ...
```

### Contents of each experiment folder


| File / folder                             | Description                                                                |
| ----------------------------------------- | -------------------------------------------------------------------------- |
| `config.json`                             | Full CLI snapshot at train time (reproducibility / methods)                |
| `results.json`                            | Best val F1, test metrics, confusion matrix, per-epoch curves, `fps`, etc. |
| `best_model.pth`                          | `state_dict` at **highest val F1** (Grad-CAM, deployment)                  |
| `last_model.pth`                          | Weights from the **last epoch** (comparison or resume)                     |
| `tensorboard/` or `events.out.tfevents.*` | TensorBoard scalars (loss, accuracy, F1, …)                                |


With `**--fold`**, `train.py` adds `**fold_0` … `fold_{K-1}**` under the experiment name; each fold stores the files above for use with `scripts/analyze_cv5_stats.py`.

### Usage notes

- **Plots / stats only**: `results.json` is enough for `scripts/analyze_results_v3.py`, `scripts/analyze_cv5_stats.py`, etc.
- **Attention visualization or deployment**: ensure `best_model.pth` exists; otherwise retrain with `train.py` using the same `config.json`, or place your checkpoint at the same relative path and run `scripts/gradcam_vis.py`.
- **Public repo size**: large files (`.pth`, raw event logs) can go via Git LFS, cloud links, or Releases; this README only documents **standard artifact names** and roles.

---

## `experiments/`: batch scripts

All scripts assume execution from the **repo root** in a **Unix-like** shell (Git Bash, WSL, Linux), with `**DATA_ROOT`** pointing at an SDNET2018-style root (default placeholder `../dataset/DATA_Maguire_20180517_ALL`—**replace with your local path**).

Environment variables (supported where noted):


| Variable                        | Meaning                                            | Example                              |
| ------------------------------- | -------------------------------------------------- | ------------------------------------ |
| `DATA_ROOT`                     | Data root (contains `D/`, `P/`, `W/`)              | `/path/to/DATA_Maguire_20180517_ALL` |
| `EPOCHS` / `BATCH` / `IMG_SIZE` | Epochs, batch size, input resolution               | `40`, `64`, `224`                    |
| `GPU`                           | CUDA device id                                     | `0`                                  |
| `PYTHON`                        | Python interpreter (`run_reviewer_ablation_v3.sh`) | `$ROOT/.venv/bin/python`             |


### Script overview


| Script                        | Default output    | Purpose                                                                                                                                                                                      |
| ----------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_experiments_v2.sh`       | `./runs_v2`       | **Full V2**: three-backbone CE baselines; MobileNetV3 robust aug / Focal / CBAM ablations; CA vs CBAM; full ResNet/EfficientNet configs; multi-GPU `&` + `wait`; ends with complexity script |
| `run_reviewer_ablation_v3.sh` | `./runs_v3`       | **Reviewer matrix**: CE / WCE / robust aug / Focal-only / CBAM-only; Focal γ sweep (1/2/3/5); matches `rev_*` names in the paper                                                             |
| `run_cv5_experiments.sh`      | `./runs_cv5`      | **Stratified 5-fold**: fixed `cv_seed`, paired `cv5_resnet18_baseline` (CE + standard pipeline) vs `cv5_resnet18_full_cbam` (robust aug + Focal + CBAM + `--no_weighted_sampler`)            |
| `generate_all_v3.sh`          | reads `./runs_v3` | **Post-process**: main tables/curves on existing `runs_v3`, `gradcam_vis.py` (default Baseline vs Ours folder names match the train scripts)                                                 |


### Experiment logic per script (paper cross-reference)

`**run_experiments_v2.sh`**

1. **Experiment 1**: ResNet18, MobileNetV3, EfficientNet under **CE + WeightedRandomSampler + standard aug** (multi-GPU parallel).
2. **Experiment 2**: MobileNetV3 ablations—robust aug only (still CE + sampler); robust aug + Focal (**sampler off**); plus CBAM (full branch).
3. **Experiment 2b**: MobileNetV3 + CA (full variant); ResNet18 / EfficientNet with **robust aug + Focal + CBAM** for cross-backbone checks.
4. **Experiment 3**: runs `scripts/complexity.py` → `runs_v2/complexity`.

`**run_reviewer_ablation_v3.sh`**

Pairs addressing typical reviewer questions: baseline CE, inverse-frequency weighted CE (`--no_weighted_sampler`), CE + robust aug, Focal without robust aug, CE + CBAM, and **γ = 1,2,3,5** sensitivity under **Focal + CBAM + robust aug** (α=0.75 fixed). Closing `echo` shows a two-terminal parallel rerun example (different `--gpu`).

`**run_cv5_experiments.sh`**

Matches the core Exp2 pairing in the main text: same fold and split seed, baseline then full method for fold-wise summaries/tests in `scripts/analyze_cv5_stats.py`.

`**generate_all_v3.sh**`

Assumes V3 training finished with `exp1_resnet18_baseline` and `exp2_ablation_full_cbam` (same naming as `runs/runs_v3` here). If your outputs live under `runs/runs_v3`, change `RUNS=./runs_v3` to `RUNS=./runs/runs_v3` inside the script, or pass the actual path as `--runs_dir`.

Example (repo root):

```bash
bash experiments/run_reviewer_ablation_v3.sh
bash experiments/run_cv5_experiments.sh
```

---

## Environment and installation

Python 3.10+ and CUDA PyTorch are recommended (match the paper setup if possible).

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

For one-off use from the repo root only, you can skip `pip install -e .` and rely on the repo root being on `sys.path` (scripts already add it).

---

## Data preparation

Scripts assume an **SDNET2018 / Maguire**-style root, e.g.:

```text
DATA_ROOT/
  D/   # Deck: .jpg under subfolders C* / U*
  P/   # Pavement
  W/   # Wall
```

Pass `--data_root` to that root; splitting logic is in `split_dataset` / `split_dataset_kfold` in `attxnet/dataset.py`.

Optional script for class ratios and split stats:

```bash
python scripts/summarize_dataset_domain.py --data_root /path/to/DATA_ROOT
python scripts/summarize_dataset_domain.py --data_root /path/to/DATA_ROOT --save_fig ./runs_v3/figures
```

---

## Training (single run)

Minimal example (SDNET **Deck only**, category `D`):

```bash
python train.py \
  --data_root /path/to/DATA_ROOT \
  --categories D \
  --backbone resnet18 \
  --attention cbam \
  --loss focal \
  --focal_alpha 0.75 --focal_gamma 2.0 \
  --use_robust_aug --no_weighted_sampler \
  --epochs 40 --batch_size 64 --img_size 224 \
  --gpu 0 \
  --output_dir ./runs_v3 \
  --exp_name my_exp
```

Notes:

- **Focal Loss**: usually add `**--no_weighted_sampler`** to avoid double reweighting with `WeightedRandomSampler`.
- **Stratified K-fold**: `--fold K --n_folds 5 --cv_seed 42` (same as `experiments/run_cv5_experiments.sh`).
- Full CLI: `python train.py --help`.

---

## Analysis and figures

Run from the repo root. If using bundled `**runs/runs_v3`**, set `--runs_dir` accordingly:

```bash
# V3 summary tables + basic curves
python scripts/analyze_results_v3.py --runs_dir ./runs/runs_v3 --output_dir ./runs/runs_v3/figures

# If outputs live directly under ./runs_v3/
# python scripts/analyze_results_v3.py --runs_dir ./runs_v3 --output_dir ./runs_v3/figures

# Numbered paper figures fig1–fig10 (default: runs_dir/figures_pub)
python scripts/plot_publication_figures.py --runs_dir ./runs/runs_v3

# Complexity (Params / FLOPs / FPS)
python scripts/complexity.py --gpu 0 --output_dir ./runs/runs_v3/complexity

# Grad-CAM (needs checkpoint dirs)
python scripts/gradcam_vis.py \
  --data_root /path/to/DATA_ROOT \
  --baseline_dir ./runs/runs_v3/exp1_resnet18_baseline \
  --ours_dir ./runs/runs_v3/exp2_ablation_full_cbam \
  --output_dir ./runs/runs_v3/figures/gradcam \
  --gpu 0

# 5-fold stats tests (needs scipy)
python scripts/analyze_cv5_stats.py --runs_dir ./runs/runs_cv5

# Reviewer Markdown + figures → outputs/revision/
python scripts/export_revision_paper_assets.py

# rev_* confusion matrices etc.
python scripts/aggregate_rev_results.py
```

(Keep `./runs/runs_v3` consistent with your clone; `gradcam` `--baseline_dir` / `--ours_dir` must point to folders that contain `best_model.pth`.)

---

## Citation

If you use this repo, cite the original paper and link this repository.

---

## License

Released under MIT by default (see `pyproject.toml`)