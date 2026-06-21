# MSFUNet: Lightweight Sow Farrowing Detection Based on Bidirectional Scoring and Multi-scale Gated Fusion

---

Authors: Chun-Cheng Weng, Ren-Shiou Liu 
Institute of Information Management, National Cheng Kung University, Tainan, 701401, Taiwan (R.O.C.)

## Description

Timely recognition of sow farrowing events is important in precision livestock farming, but practical monitoring systems often rely on fixed cameras and limited computational resources. This study formulates the visible emergence of a newborn piglet from the birth canal as frame-level binary classification between `Exposed` and `Not Exposed`.

To address small target regions, occlusion, and illumination variation in surveillance images, this study proposes MSFUNet, a lightweight SqueezeNet-based model with multi-scale feature fusion, bidirectional query scoring, gated residual fusion, and guided pooling. The repository provides a controlled leave-one-pig-out evaluation workflow for measuring recognition performance and computational efficiency on 48,000 labeled images from 8 sows.

This repository contains the code used to reproduce six experiments. The experiments are designed to answer:

- whether the improvement comes from the proposed MSFUNet design;
- how MSFUNet compares with other backbone models under LOPO evaluation;
- whether the dual-score weighting mechanism is necessary;
- which feature levels contribute most to F1, Recall, and AUC;
- why Pool3 + Pool5 + Fire9 is selected as the standard design;
- whether the model is efficient enough for deployment.

---

## Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Experimental Protocol](#experimental-protocol)
- [Results](#results)
- [Usage](#usage)
- [Experiments](#experiments)
- [Experiment 1: SqNet, Fusion Only, and MSFUNet Full](#experiment-1-sqnet-fusion-only-and-msfunet-full)
- [Experiment 2: Backbone Model Comparison](#experiment-2-backbone-model-comparison)
- [Experiment 3: Weight Source Design](#experiment-3-weight-source-design)
- [Experiment 4: Progressive Feature-Level Ablation](#experiment-4-progressive-feature-level-ablation)
- [Experiment 5: Feature Extraction Level Selection](#experiment-5-feature-extraction-level-selection)
- [Experiment 6: Model Efficiency and Deployment Feasibility](#experiment-6-model-efficiency-and-deployment-feasibility)

---

## Installation

### Prerequisites

- Python 3.9
- PyTorch
- torchvision
- CUDA-enabled GPU is recommended for training and efficiency benchmarking
- Additional dependencies listed in `requirements.txt`

### Setup

Clone the repository:

```bash
git clone https://github.com/rsliu/ccweng-sow-farrowing-detection.git
cd ccweng-sow-farrowing-detection
```

Create and activate an environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the main dependencies:

```bash
pip install -r requirements.txt
```

If the project is already running inside an existing Conda environment, activate that environment instead of creating a new one.

### GPU

Model training and efficiency benchmarking are intended to run on a CUDA-enabled GPU. Before running the experiments, check GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, PyTorch can access the GPU. Full LOPO training and latency/FPS benchmarking should be run on GPU for practical runtime and comparable efficiency results.

## Dataset

E1–E5 use pig-level LOPO data:

```text
Dataset/lopo/
  exposed/
    pig01/
      image_001.jpg
      image_002.jpg
    pig02/
      image_001.jpg
  not_exposed/
    pig01/
      image_001.jpg
    pig02/
      image_001.jpg
```

E6 uses a flat binary ImageFolder dataset:

```text
Dataset/full/
  exposed/
    image_001.jpg
  not_exposed/
    image_001.jpg
```

After placement, the project root should contain both datasets:

```text
ccweng-sow-farrowing-detection/
  Dataset/
    lopo/
      exposed/
        pig01/
      not_exposed/
        pig01/
    full/
      exposed/
      not_exposed/
  train.py
  README.md
```

The entry point selects `Dataset/lopo` for E1–E5 and `Dataset/full` for E6. No
dataset argument is required when these default paths are used.

If the dataset is stored in another location, pass it with `--data_root`:

```bash
python train.py \
  --case E1_msfunet_full \
  --data_root /path/to/Dataset/lopo \
  --run
```

The default ROI configuration path is:

```text
config/roi_pig.json
```

To specify another ROI configuration file:

```bash
python train.py \
  --case E1_msfunet_full \
  --data_root Dataset/lopo \
  --roi_cfg /path/to/roi_pig.json \
  --run
```

## Project Structure

```text
ccweng-sow-farrowing-detection/
  train.py                       # Main entry point for all experiment cases
  README.md                      # Setup, experiment, and output guide
  requirements.txt              # Reference-environment package versions
  code/
    models/
      __init__.py                # Public model exports
      msfunet.py                 # Proposed model: MSFUNet
      comparison_models.py       # Comparison models: SqNet baseline, MSANet
      layers.py                  # Shared layers
      factory.py                 # build_model entry used by trainers
      _squeezenet_core.py        # Shared internal implementation
    utils/
      __init__.py                # Public utility exports
      data.py                    # Dataset path, ROI, and letterbox helpers
      metrics.py                 # Metric helpers
      benchmark.py               # Parameter count and model-size helpers
      seed.py                    # Random seed helpers
    trainers/
      msfunet_ablation/
        lopo_msfunet_ablation_trainer.py  # Shared E1/E3/E4/E5 LOPO pipeline
      ex2_backbone_comparison/
        lopo_squeezenet_baseline_trainer.py  # SqueezeNet comparison
        lopo_resnet18_trainer.py             # ResNet-18 comparison
        lopo_vit_tiny_trainer.py             # ViT-Tiny comparison
        lopo_vit_base_trainer.py             # ViT-Base comparison
        lopo_msanet_trainer.py                # MSANet comparison
      ex6_efficiency/
        benchmark_cnn_models_efficiency.py    # CNN benchmarks
        benchmark_vit_tiny_efficiency.py      # ViT-Tiny benchmark
        benchmark_vit_base_efficiency.py      # ViT-Base benchmark
        measure_resnet18_efficiency.py         # ResNet-18 measurements
    experiments.py               # Experiment cases and paths
    reproducibility.py           # Dataset preflight and output-status checks
    run.py                       # Command construction and group execution
  config/
    roi_pig.json                  # Normalized per-pig ROI coordinates
```

## Experimental Protocol

The six experiment groups use two different dataset layouts. They must not be
interchanged:

| Experiments | Protocol | Default data root | Required layout |
|---|---|---|---|
| E1–E5 | 8-fold leave-one-pig-out (LOPO) | `Dataset/lopo` | `<class>/<pig_id>/<image>` |
| E6 | Stratified efficiency benchmark | `Dataset/full` | `<class>/<image>` |

The unified entry point selects the correct root automatically. Before printing
or executing a command it checks the class folders, image counts and, for LOPO,
the pig folders. A flat dataset therefore cannot accidentally be used for E1–E5.

The workflow uses the architecture-focused paper protocol by default.
Its controlled settings include seed 42, 55 epochs, ROI/letterbox preprocessing,
weighted sampling, EMA, temperature scaling, and threshold selection using only
the validation split. No test labels are used for model selection or threshold
tuning. The proposed architecture is evaluated consistently across all relevant
ablation cases.

The entry point fixes `PYTHONHASHSEED`, enables deterministic PyTorch algorithms,
disables cuDNN autotuning and TF32 for LOPO training, and sets the cuBLAS
deterministic workspace configuration. Exact floating-point identity still
requires the same locked software environment, GPU model, driver, and input data.

The organized trainers retain the original model and optimization logic.
Organizational changes are limited to source-reference headers, import-path
bootstrapping, professional comment typography, explicit output paths, dataset
preflight validation, and standardized experiment registration.

The experiment workflow is self-contained: MSFUNet cases load
`code/models/factory.py` rather than depending on external development copies
of the model files.

Before training, check the configuration and dataset:

```bash
python train.py --audit
```

The check summarizes the registered configuration and available outputs. A run
is `COMPLETE` only when every expected result file exists, contains the required
metric columns, and includes the minimum fold coverage.

For the current workspace, the entry point automatically prefers `bin/python`
so the installed experiment environment is used even when the shell has not
been activated. On another machine, activate the intended environment first.

### Metrics and outputs

For binary LOPO experiments, positive class means `Exposed` and each fold writes
Accuracy, Precision, Recall, F1, Specificity and ROC-AUC. Thresholds are selected
using validation data and then carried to the held-out test pig. The standard
outputs are:

- per-fold training history and test result: `foldNN_<pig>.csv/.txt` or the
  backbone trainer's equivalent `lopo_<pig>.csv/.txt`;
- per-pig and merged raw/row-normalized confusion matrices;
- `summary.csv`, `lopo_summary.csv`, or the trainer-specific summary file;
- `config.json` and per-pig best/final checkpoints where supported.

Experiment outputs are stored by case:

```text
Result/runs/<case_key>/          # Metrics, logs, and confusion matrices
Model/checkpoints/<case_key>/    # Checkpoints
```

| Group | Evaluation | Principal metrics | Command |
|---|---|---|---|
| E1 | SqNet/Fusion/MSFUNet comparison | Accuracy, Precision, Recall, F1, Specificity, AUC | `python train.py --group E1 --run` |
| E2 | Backbone comparison | Accuracy, Precision, Recall, F1, Specificity, AUC, parameters, latency, FPS | `python train.py --group E2 --run` |
| E3 | Weight-source ablation | Accuracy, Precision, Recall, F1, Specificity, AUC | `python train.py --group E3 --run` |
| E4 | Progressive feature ablation | Accuracy, Precision, Recall, F1, Specificity, AUC | `python train.py --group E4 --run` |
| E5 | Feature-level selection | Accuracy, Precision, Recall, F1, Specificity, AUC | `python train.py --group E5 --run` |
| E6 | Efficiency benchmark | Model-specific parameters, latency, FPS, FLOPs, and serialized size | `python train.py --group E6 --run` |

E6 reports parameter count, latency and FPS using the original benchmark method;
model-specific scripts additionally report their available FLOPs and serialized
model size. Latency/FPS are hardware measurements and are comparable only when
device, CUDA/cuDNN, image size, warm-up, iterations and batch size are identical.

## Results

The following tables summarize the principal result.
Classification metrics are the mean results of eight-fold LOPO evaluation.
Accuracy, Precision, Recall, F1-score and Specificity are reported as percentages.

### Classification performance

| Model | Accuracy | Precision | Recall | F1-score | Specificity | AUC |
|---|---:|---:|---:|---:|---:|---:|
| SqueezeNet | 63.08 | 61.27 | 74.42 | 64.60 | 51.73 | 0.7170 |
| ResNet-18 | 56.38 | 54.08 | 39.55 | 39.94 | **73.20** | 0.6167 |
| ViT-Tiny | 61.70 | 63.99 | 74.63 | 65.90 | 48.77 | 0.6110 |
| ViT-Base | 52.61 | 42.43 | 62.56 | 49.36 | 42.66 | 0.5330 |
| MSANet | 63.74 | 61.09 | **82.69** | 67.94 | 44.78 | 0.7208 |
| **MSFUNet** | **69.98** | **66.79** | 79.44 | **72.60** | 60.52 | **0.7565** |

### Model efficiency

Efficiency was measured using single-image inference on an NVIDIA RTX 4080.
Latency and FPS are hardware-dependent and are intended for relative comparison.

| Model | F1-score | Params (M) | FLOPs (G) | Latency (ms) | FPS | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| SqueezeNet | 64.60 | **0.72** | **0.26** | **0.71** | **1409.13** | **2.79** |
| ResNet-18 | 39.94 | 11.178 | 1.824 | 0.778 | 1285.58 | 42.72 |
| ViT-Tiny | 65.90 | 5.525 | 1.078 | 2.018 | 495.61 | 21.21 |
| ViT-Base | 49.36 | 85.8 | 16.863 | 3.180 | 314.45 | 327.44 |
| MSANet | 67.94 | 1.45 | 0.55 | 1.41 | 710.96 | 5.58 |
| **MSFUNet** | **72.60** | 2.46 | 0.46 | 1.32 | 755.47 | 9.42 |

## Usage

### List All Experiment Cases

```bash
python train.py --list
```

### Preview an Experiment Command

```bash
python train.py --case E1_msfunet_full
```

### Run an Experiment

The default LOPO dataset is selected automatically. Run:

```bash
python train.py \
  --case E1_msfunet_full \
  --data_root Dataset/lopo \
  --roi_cfg config/roi_pig.json \
  --epochs 55 \
  --batch 16 \
  --num_workers 4 \
  --img_size 224 \
  --run
```

Run or preview every registered case in one experiment group:

```bash
# Dry run: validates data and prints every E1 command.
python train.py --group E1

# Execute every E1 case sequentially.
python train.py --group E1 --run
```

`--group E6` automatically expands the CNN benchmark into `squeezenet`,
`msanet35`, `msanet53`, and `msfu`, followed by ViT-Tiny, ViT-B, and ResNet-18.
Use the same GPU and an otherwise idle machine for the entire E6 group.

If the dataset is placed elsewhere, change `--data_root` to that dataset path.

## Experiments

### Experiment 1: SqNet, Fusion Only, and MSFUNet Full

Purpose: verify whether the performance gain comes from the proposed MSFUNet design.

Cases:

- `E1_baseline_squeezenet`
- `E1_fusion_only`
- `E1_msfunet_full`

Example:

```bash
python train.py --case E1_msfunet_full --run
```

### Experiment 2: Backbone Model Comparison

Purpose: compare MSFUNet with representative backbone and comparison models under LOPO evaluation.

Cases:

- `E2_squeezenet_lopo`
- `E2_resnet18_lopo`
- `E2_vit_tiny_lopo`
- `E2_vit_base_lopo`
- `E2_msanet_lopo`
- `E2_msfunet_full`

Example:

```bash
python train.py --case E2_resnet18_lopo --run
```

### Experiment 3: Weight Source Design

Purpose: analyze whether Deep-only, Shallow-only, or Dual-score weighting gives better recognition performance.

Cases:

- `E3_deep_only`
- `E3_shallow_only`
- `E3_dual_learnable`

Example:

```bash
python train.py --case E3_dual_learnable --run
```

### Experiment 4: Progressive Feature-Level Ablation

Purpose: evaluate the contribution of shallow and middle features to F1, Recall, and AUC.

Cases:

- `E4_pool3_fire9`
- `E4_pool5_fire9`
- `E4_pool3_pool5_fire9_full`

Example:

```bash
python train.py --case E4_pool3_pool5_fire9_full --run
```

### Experiment 5: Feature Extraction Level Selection

Purpose: verify why Pool3 + Pool5 + Fire9 is selected as the standard MSFUNet design.

Cases:

- `E5I_fire3_pool5_fire9`
- `E5I_fire4_pool5_fire9`
- `E5_standard_pool3_pool5_fire9`
- `E5II_pool3_fire5_fire9`
- `E5II_pool3_fire6_fire9`
- `E5II_pool3_fire7_fire9`

Example:

```bash
python train.py --case E5_standard_pool3_pool5_fire9 --run
```

### Experiment 6: Model Efficiency and Deployment Feasibility

Purpose: compare parameter count, FLOPs, latency, FPS, and model size.

Cases:

- `E6_cnn_models_efficiency`: SqNet, MSANet, and MSFUNet
- `E6_vit_tiny_efficiency`
- `E6_vit_base_efficiency`
- `E6_resnet18_flops`

Run CNN efficiency benchmark:

```bash
python train.py \
  --case E6_cnn_models_efficiency \
  --efficiency_model msfu \
  --run
```

Run the same case with other CNN comparison models:

```bash
python train.py --case E6_cnn_models_efficiency --efficiency_model squeezenet --run
python train.py --case E6_cnn_models_efficiency --efficiency_model msanet35 --run
python train.py --case E6_cnn_models_efficiency --efficiency_model msanet53 --run
```

Run ViT and ResNet-18 efficiency cases:

```bash
python train.py --case E6_vit_tiny_efficiency --run
python train.py --case E6_vit_base_efficiency --run
python train.py --case E6_resnet18_flops --run
```

For strict efficiency comparison, run all efficiency cases on the same device with the same batch size and image size.

After completing the full GPU runs, check the outputs again:

```bash
python train.py --audit
```
