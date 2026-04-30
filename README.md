# AttXNet

面向桥梁（Infrastructure）场景的表面裂缝图像二分类代码：**骨干网络（ResNet18 / MobileNetV3 / EfficientNet-B0）** + **可选注意力（CBAM / CA）**，配合 **交叉熵 / 加权 CE / Focal Loss**、**稳健增强**与 **分层划分 / K 折**，适用于 SDNET2018 风格的 `{D,P,W}` 目录布局。

本仓库在 `runs/` 下保留了作者实际跑过的实验产物（指标 JSON、配置、TensorBoard 日志等；**权重文件 `*.pth` 视分发体积可能未纳入版本库**——若目录中缺少权重，用相同 `config.json` 复训即可得到一致流程）。`experiments/` 提供与论文 / 审稿对应的 Bash 批处理入口，便于一键复现实验矩阵。

**请在仓库根目录（本 README 所在目录）打开终端再执行命令**，以保证 `attxnet` 包、`train.py` 与 `--output_dir` 相对路径一致。

---

## 仓库结构

| 路径 | 说明 |
| :--- | :--- |
| `attxnet/` | 可导入核心库：`dataset`、`models`、`losses` |
| `train.py` | 训练与测试入口：写入 `results.json`、TensorBoard、`best_model.pth` / `last_model.pth` |
| `scripts/` | 复杂度统计、Grad-CAM、结果汇总、`rev_*` 聚合与论文级图表导出 |
| `experiments/` | **批量实验** Bash 脚本（见下文专节）；需在 **Git Bash / Linux / WSL** 下执行 |
| `runs/` | **历史实验输出根目录**（见下文）：默认包含 `runs_v3/`（主实验 + 审稿消融）、`runs_cv5/`（五折配对）等 |
| `outputs/revision/` | 运行 `scripts/export_revision_paper_assets.py` 后生成的审稿补充材料 |

脚本里常把 `--output_dir` 设为 `./runs_v3` 或 `./runs_cv5`（仓库根目录下）；本仓库同时将一批已跑结果汇总在 **`runs/runs_v3`、`runs/runs_cv5`**。**运行分析脚本时，`--runs_dir` 请指向你实际存放 `results.json` 的那一层**（例如 `./runs/runs_v3` 或 `./runs_v3`）。

---

## `runs/`：实验产物与权重

`runs/` 用于集中存放每次训练产生的目录，便于对照论文表格与复现图表。典型布局如下（以本仓库已提交的内容为例）：

```text
runs/
  runs_v3/                    # 单次训练 / 消融 / 跨骨干 / 审稿补充
    exp1_resnet18_baseline/
    exp2_ablation_full_cbam/
    rev_baseline_ce/
    rev_focal_gamma_2/
    ...
    figures/                  # 由 scripts 生成的表、曲线、公开图等
  runs_cv5/                   # 分层 5 折交叉验证（配对对比）
    cv5_resnet18_baseline/
      fold_0/
      fold_1/
      ...
    cv5_resnet18_full_cbam/
      ...
```

### 每个实验子目录里有什么

| 文件 / 目录 | 说明 |
| :--- | :--- |
| `config.json` | 训练时的完整 CLI 配置快照，复现或写方法学时的依据 |
| `results.json` | 验证集最优 F1、测试集指标、混淆矩阵、各 epoch 曲线、`fps` 等 |
| `best_model.pth` | 验证集 **F1 最高** 时的 `state_dict`（下游 Grad-CAM、部署推理可直接加载） |
| `last_model.pth` | **最后一个 epoch** 的权重，用于对照或继续训练 |
| `tensorboard/` 或 `events.out.tfevents.*` | TensorBoard 标量曲线（损失、准确率、F1 等） |

启用 **`--fold`** 时，`train.py` 会在实验名目录下再建 **`fold_0` … `fold_{K-1}`**，每个 fold 各自保存上述文件，便于与 `scripts/analyze_cv5_stats.py` 汇总对接。

### 使用说明

- **只做图表 / 统计**：有 `results.json` 即可运行 `scripts/analyze_results_v3.py`、`scripts/analyze_cv5_stats.py` 等。
- **需要可视化注意力或部署**：请确认对应目录下存在 `best_model.pth`；若无，可根据同目录 `config.json` 执行 `train.py` 重新训练，或将你自己的 checkpoint 放到相同相对路径后运行 `scripts/gradcam_vis.py`。
- **公开仓库体积**：大文件（`.pth`、原始事件文件）可选用 Git LFS、网盘链接或 Release 附件分发；本 README 仅说明**标准产物命名**与用途。

---

## `experiments/`：批量实验脚本详解

所有脚本均假定：在 **类 Unix 环境**（Git Bash、WSL、Linux）下，于**仓库根目录**执行；并通过变量 **`DATA_ROOT`** 指向 SDNET2018 风格数据根（脚本内默认为占位路径 `../dataset/DATA_Maguire_20180517_ALL`，**务必改为你本机路径**）。

环境变量（部分脚本支持）：

| 变量 | 含义 | 示例 |
| :--- | :--- | :--- |
| `DATA_ROOT` | 数据根（含 `D/`、`P/`、`W/`） | `/path/to/DATA_Maguire_20180517_ALL` |
| `EPOCHS` / `BATCH` / `IMG_SIZE` | 训练轮数、批量、输入分辨率 | `40`、`64`、`224` |
| `GPU` | CUDA 设备号 | `0` |
| `PYTHON` | Python 解释器（`run_reviewer_ablation_v3.sh`） | `$ROOT/.venv/bin/python` |

### 脚本总览

| 脚本 | 输出目录（默认） | 目的摘要 |
| :--- | :--- | :--- |
| `run_experiments_v2.sh` | `./runs_v2` | **V2 整套**：三骨干 CE 基线、MobileNetV3 上稳健增强 / Focal / CBAM 消融、CA 对比、ResNet/EfficientNet 全配置；含多 GPU 并行 `&` + `wait`；末尾调用复杂度脚本 |
| `run_reviewer_ablation_v3.sh` | `./runs_v3` | **审稿补充矩阵**：CE / WCE / 稳健增强 / 仅 Focal / 仅 CBAM、Focal 的 γ 扫描（1/2/3/5）等；与文中 `rev_*` 实验名一一对应 |
| `run_cv5_experiments.sh` | `./runs_cv5` | **分层 5 折**：固定 `cv_seed`，两配置配对——`cv5_resnet18_baseline`（CE + 标准流程）vs `cv5_resnet18_full_cbam`（稳健增强 + Focal + CBAM + `--no_weighted_sampler`） |
| `generate_all_v3.sh` | 读取 `./runs_v3` | **后处理**：在已有 `runs_v3` 上跑主结果表与曲线、`gradcam_vis.py`（默认 Baseline vs Ours 目录名与训练脚本一致） |

### 各脚本实验逻辑（便于对照论文小节）

**`run_experiments_v2.sh`**

1. **Experiment 1**：ResNet18、MobileNetV3、EfficientNet 在 **CE + WeightedRandomSampler + 标准增强** 下的骨干对比（多卡并行）。
2. **Experiment 2**：在 MobileNetV3 上做消融——仅稳健增强（仍为 CE + sampler）、稳健增强 + Focal（**关闭 sampler**）、再加 CBAM（完整分支）。
3. **Experiment 2b**：MobileNetV3 + CA（完整方法变体）；ResNet18 / EfficientNet 上跑「稳健增强 + Focal + CBAM」以跨骨干验证。
4. **Experiment 3**：调用 `scripts/complexity.py` 写入 `runs_v2/complexity`。

**`run_reviewer_ablation_v3.sh`**

按审稿人常见质疑逐项构造对照：基线 CE、逆频加权 CE（`--no_weighted_sampler`）、CE + 稳健增强、无稳健增强的 Focal、CE + CBAM、以及 **Focal + CBAM + 稳健增强** 下 **γ = 1,2,3,5** 的敏感性分析（α=0.75 固定）。脚本末尾 `echo` 给出双终端并行补跑示例（不同 `--gpu`）。

**`run_cv5_experiments.sh`**

与主文 Exp2 核心配对一致：同一折、同一划分随机种子下，先跑基线再跑完整方法，便于 `scripts/analyze_cv5_stats.py` 做折间汇总与检验。

**`generate_all_v3.sh`**

假设已完成 V3 系列训练且存在 `exp1_resnet18_baseline` 与 `exp2_ablation_full_cbam`（与仓库内 `runs/runs_v3` 命名一致）。若你的输出在 `runs/runs_v3`，请把脚本内 `RUNS=./runs_v3` 改为 `RUNS=./runs/runs_v3`，或把分析命令里的 `--runs_dir` 指向实际路径。

执行示例（仓库根目录）：

```bash
bash experiments/run_reviewer_ablation_v3.sh
bash experiments/run_cv5_experiments.sh
```

---

## 环境与安装

建议使用 Python 3.10+ 与 CUDA 版 PyTorch（与论文机器一致为佳）。

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

仅在本仓库根目录临时使用时，也可跳过 `pip install -e .`，直接依赖当前工作目录已加入 `sys.path`（脚本内已对仓库根目录做了兼容）。

---

## 数据准备

脚本假定 **SDNET2018 / Maguire** 风格根目录，例如：

```text
DATA_ROOT/
  D/   # Deck：子文件夹 C* / U* 下为 .jpg
  P/   # Pavement
  W/   # Wall
```

命令行通过 `--data_root` 指向该根目录；具体划分逻辑见 `attxnet/dataset.py` 中的 `split_dataset` / `split_dataset_kfold`。

可选用脚本打印类别占比与划分统计：

```bash
python scripts/summarize_dataset_domain.py --data_root /path/to/DATA_ROOT
python scripts/summarize_dataset_domain.py --data_root /path/to/DATA_ROOT --save_fig ./runs_v3/figures
```

---

## 训练（单次实验）

最小示例（SDNET **仅 Deck**，即 `D`）：

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

要点：

- **Focal Loss**：通常加上 **`--no_weighted_sampler`**，避免与 `WeightedRandomSampler` 双重重加权。
- **分层 K 折**：使用 `--fold K --n_folds 5 --cv_seed 42`（与 `experiments/run_cv5_experiments.sh` 一致）。
- 完整 CLI 说明：`python train.py --help`。

---

## 结果分析与制图

均在仓库根目录执行。若使用本仓库自带的 **`runs/runs_v3`**，请将 `--runs_dir` 设为该路径：

```bash
# V3 总表 + 基础曲线
python scripts/analyze_results_v3.py --runs_dir ./runs/runs_v3 --output_dir ./runs/runs_v3/figures

# 若输出直接在仓库根下的 runs_v3/
# python scripts/analyze_results_v3.py --runs_dir ./runs_v3 --output_dir ./runs_v3/figures

# 编号论文图 fig1–fig10（默认写入 runs_dir/figures_pub）
python scripts/plot_publication_figures.py --runs_dir ./runs/runs_v3

# 复杂度（Params / FLOPs / FPS）
python scripts/complexity.py --gpu 0 --output_dir ./runs/runs_v3/complexity

# Grad-CAM（需已有 checkpoint 目录）
python scripts/gradcam_vis.py \
  --data_root /path/to/DATA_ROOT \
  --baseline_dir ./runs/runs_v3/exp1_resnet18_baseline \
  --ours_dir ./runs/runs_v3/exp2_ablation_full_cbam \
  --output_dir ./runs/runs_v3/figures/gradcam \
  --gpu 0

# 5 折统计检验（需 scipy）
python scripts/analyze_cv5_stats.py --runs_dir ./runs/runs_cv5

# 审稿 Markdown + 图 → outputs/revision/
python scripts/export_revision_paper_assets.py

# rev_* 汇总混淆矩阵等
python scripts/aggregate_rev_results.py
```

（如果你克隆后的路径是 `runs/runs_v3`，把上述 `./runs/runs_v3` 保持与本地一致即可；`gradcam` 的 `--baseline_dir` / `--ours_dir` 需指向真实存在 `best_model.pth` 的子目录。）

---

## 引用

若使用本仓库，请在论文中引用原文并附上仓库链接（发表后可将此处替换为 DOI / Zenodo）。

---

## 许可证

默认以 MIT 形式公开（见 `pyproject.toml`）；若单位另有要求，请自行替换许可条款。
