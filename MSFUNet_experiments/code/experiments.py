# -*- coding: utf-8 -*-
"""Experiment registry for the organized MSFUNet experiments.

The registry is the single place that maps paper experiments to:

- model variant
- model kwargs
- trainer script
- result/model output directories
- existing result directories used in the current project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ExperimentCase:
    key: str
    title: str
    trainer: str
    result_dir: str
    model_dir: str
    model_kwargs: Dict[str, object] = field(default_factory=dict)
    model_py: Optional[str] = None
    default_data_root: str = "Dataset/lopo"
    data_layout: str = "lopo"
    metrics: tuple = ("Accuracy", "Precision", "Recall", "F1", "Specificity", "AUC")
    existing_result_dir: Optional[str] = None
    notes: str = ""


GENERIC_TRAINER = "MSFUNet_experiments/code/trainers/msfunet_ablation/lopo_msfunet_ablation_trainer.py"
UNIFIED_MODEL = "MSFUNet_experiments/code/models/factory.py"

# Values recorded in the original E1/E3/E4/E5 result config.json files.  Keep
# them explicit here: changing a model default must not silently change a paper
# generated command.
LEGACY_MSFU_DEFAULTS = {
    "topk_ratio": 0.0,
    "style_p": 0.0,
    "style_alpha": 0.0,
    "use_style_norm": True,
    "msfu_bg_scale": 0.0,
    "msfu_init_gamma": 0.05,
    "softk_tau": 0.5,
    "softk_alpha": 1.0,
    "use_coord_score": True,
    "use_local_refine": True,
    "qk_dim": 128,
    "use_se_in_fusion": True,
}


EXPERIMENTS: Dict[str, List[ExperimentCase]] = {
    "E1": [
        ExperimentCase(
            key="E1_baseline_squeezenet",
            title="SqueezeNet baseline with GAP",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E1-1/Baseline_SqueezeNet_GAP",
            model_dir="Model/E1-1/Baseline_SqueezeNet_GAP",
            existing_result_dir="Result/E1-1/Baseline_SqueezeNet_GAP",
            model_py="Model/squeezenet_msfu_ex1.py",
            model_kwargs={
                "exp_variant": "baseline", "msfu_mode": "full", "tap_idx_z": 5, "tap_idx_y": 8,
                "pool_type": "gap", "topk_ratio": 0.0, "style_p": 0.0, "style_alpha": 0.0,
                "use_style_norm": True, "msfu_bg_scale": 0.0, "msfu_init_gamma": 0.05,
                "softk_tau": 0.5, "softk_alpha": 1.0, "use_coord_score": True,
                "use_local_refine": True, "qk_dim": 128, "use_se_in_fusion": True,
            },
        ),
        ExperimentCase(
            key="E1_fusion_only",
            title="MSFUNet fusion-only without BQS/guided pooling",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E1-2/MSFU_NoW_FusionOnly_GAP",
            model_dir="Model/E1-2/MSFU_NoW_FusionOnly_GAP",
            existing_result_dir="Result/E1-2/MSFU_NoW_FusionOnly_GAP",
            model_py="Model/squeezenet_msfu_ex1.py",
            model_kwargs={
                **LEGACY_MSFU_DEFAULTS,
                "exp_variant": "msfu",
                "msfu_mode": "now",
                "tap_idx_z": 5,
                "tap_idx_y": 8,
                "pool_type": "gap",
                "topk_ratio": 0.0, "style_p": 0.0, "style_alpha": 0.0,
                "use_style_norm": True, "msfu_bg_scale": 0.0, "msfu_init_gamma": 0.05,
                "softk_tau": 0.5, "softk_alpha": 1.0, "use_coord_score": True,
                "use_local_refine": True, "qk_dim": 128, "use_se_in_fusion": True,
            },
        ),
        ExperimentCase(
            key="E1_msfunet_full",
            title="Full MSFUNet with dual-score BQS and guided pooling",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E1-3/MSFU_Full_BQS_Guided",
            model_dir="Model/E1-3/MSFU_Full_BQS_Guided",
            existing_result_dir="Result/E1-3/MSFU_Full_BQS_Guided",
            model_py="Model/squeezenet_msfu_ex1.py",
            model_kwargs={
                **LEGACY_MSFU_DEFAULTS,
                "exp_variant": "msfu",
                "msfu_mode": "full",
                "tap_idx_z": 5,
                "tap_idx_y": 8,
                "pool_type": "guided",
                "topk_ratio": 0.0, "style_p": 0.0, "style_alpha": 0.0,
                "use_style_norm": True, "msfu_bg_scale": 0.0, "msfu_init_gamma": 0.05,
                "softk_tau": 0.5, "softk_alpha": 1.0, "use_coord_score": True,
                "use_local_refine": True, "qk_dim": 128, "use_se_in_fusion": True,
            },
        ),
    ],
    "E2": [
        ExperimentCase(
            key="E2_squeezenet_lopo",
            title="SqueezeNet LOPO backbone baseline",
            trainer="MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_squeezenet_baseline_trainer.py",
            result_dir="Result/squeezenet_lopo_v1",
            model_dir="Model/squeezenet_lopo_v1",
            existing_result_dir="Result/squeezenet_lopo_v1",
            notes="Backbone comparison: SqNet.",
        ),
        ExperimentCase(
            key="E2_resnet18_lopo",
            title="ResNet-18 LOPO backbone baseline",
            trainer="MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_resnet18_trainer.py",
            result_dir="Result/lopo_resnet18",
            model_dir="Model/lopo_resnet18",
            existing_result_dir="Result/lopo_resnet18",
            notes="Backbone comparison: ResNet-18.",
        ),
        ExperimentCase(
            key="E2_vit_tiny_lopo",
            title="ViT-Tiny LOPO backbone baseline",
            trainer="MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_vit_tiny_trainer.py",
            result_dir="Result/lopo_vit_timm_unified",
            model_dir="Model/lopo_vit_timm_unified",
            existing_result_dir="Result/lopo_vit_timm_unified",
            notes="Backbone comparison: timm vit_tiny_patch16_224.",
        ),
        ExperimentCase(
            key="E2_vit_base_lopo",
            title="ViT-B LOPO backbone baseline",
            trainer="MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_vit_base_trainer.py",
            result_dir="Result/vit_lopo_v1",
            model_dir="Model/vit_lopo_v1",
            existing_result_dir="Result/vit_lopo_v1",
            notes="Backbone comparison: HuggingFace ViT-B.",
        ),
        ExperimentCase(
            key="E2_msanet_lopo",
            title="MSANet LOPO comparison model",
            trainer="MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_msanet_trainer.py",
            result_dir="Result/msa_v1",
            model_dir="Model/msa_v1",
            existing_result_dir="Result/msa_v1",
            notes="Backbone comparison: original MSANet.",
        ),
        ExperimentCase(
            key="E2_msfunet_full",
            title="MSFUNet full model",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E1-3/MSFU_Full_BQS_Guided",
            model_dir="Model/E1-3/MSFU_Full_BQS_Guided",
            existing_result_dir="Result/E1-3/MSFU_Full_BQS_Guided",
            model_py="Model/squeezenet_msfu_ex1.py",
            model_kwargs={
                **LEGACY_MSFU_DEFAULTS,
                "exp_variant": "msfu",
                "msfu_mode": "full",
                "tap_idx_z": 5,
                "tap_idx_y": 8,
                "pool_type": "guided",
            },
        ),
    ],
    "E3": [
        ExperimentCase(
            key="E3_deep_only",
            title="BQS weight source: deep-only",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E2-1/E2-1_BQS_deep_only",
            model_dir="Model/E2-1/E2-1_BQS_deep_only",
            existing_result_dir="Result/E2-1/E2-1_BQS_deep_only",
            model_py="Model/squeezenet_msfu_ex2.py",
            model_kwargs={
                **LEGACY_MSFU_DEFAULTS,
                "exp_variant": "msfu",
                "msfu_mode": "full",
                "tap_idx_z": 5,
                "tap_idx_y": 8,
                "pool_type": "guided",
                "bqs_mode": "deep",
            },
            notes="Existing folder uses old E2 numbering.",
        ),
        ExperimentCase(
            key="E3_shallow_only",
            title="BQS weight source: shallow-only",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E2-2/E2-2_BQS_shal_only",
            model_dir="Model/E2-2/E2-2_BQS_shal_only",
            existing_result_dir="Result/E2-2/E2-2_BQS_shal_only",
            model_py="Model/squeezenet_msfu_ex2.py",
            model_kwargs={
                **LEGACY_MSFU_DEFAULTS,
                "exp_variant": "msfu",
                "msfu_mode": "full",
                "tap_idx_z": 5,
                "tap_idx_y": 8,
                "pool_type": "guided",
                "bqs_mode": "shal",
            },
            notes="Existing folder uses old E2 numbering.",
        ),
        ExperimentCase(
            key="E3_dual_learnable",
            title="BQS weight source: dual-score learnable lambda",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E2-3/E2-3_BQS_dual_learnable_lambda",
            model_dir="Model/E2-3/E2-3_BQS_dual_learnable_lambda",
            existing_result_dir="Result/E2-3/E2-3_BQS_dual_learnable_lambda",
            model_py="Model/squeezenet_msfu_ex2.py",
            model_kwargs={
                **LEGACY_MSFU_DEFAULTS,
                "exp_variant": "msfu",
                "msfu_mode": "full",
                "tap_idx_z": 5,
                "tap_idx_y": 8,
                "pool_type": "guided",
                "bqs_mode": "dual",
                "lambda_mode": "learnable",
                "lambda_fixed": 0.5,
            },
            notes="Existing folder uses old E2 numbering.",
        ),
    ],
    "E4": [
        ExperimentCase(
            key="E4_pool3_fire9",
            title="Layer ablation: Pool3 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E4/E4_Pool3_Fire9",
            model_dir="Model/E4/E4_Pool3_Fire9",
            existing_result_dir="Result/E4/E4_Pool3_Fire9",
            model_py="Model/squeezenet_msfu_ex4.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_x": 12, "tap_idx_z": 5, "tap_idx_y": -1, "pool_type": "guided", "bqs_mode": "dual", "lambda_mode": "learnable", "lambda_fixed": 0.5},
        ),
        ExperimentCase(
            key="E4_pool5_fire9",
            title="Layer ablation: Pool5 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E4/E4_Pool5_Fire9",
            model_dir="Model/E4/E4_Pool5_Fire9",
            existing_result_dir="Result/E4/E4_Pool5_Fire9",
            model_py="Model/squeezenet_msfu_ex4.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_x": 12, "tap_idx_z": -1, "tap_idx_y": 8, "pool_type": "guided", "bqs_mode": "dual", "lambda_mode": "learnable", "lambda_fixed": 0.5},
        ),
        ExperimentCase(
            key="E4_pool3_pool5_fire9_full",
            title="Layer ablation: Pool3 + Pool5 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E4/E4_Pool3_Pool5_Fire9_Full",
            model_dir="Model/E4/E4_Pool3_Pool5_Fire9_Full",
            existing_result_dir="Result/E1-3/MSFU_Full_BQS_Guided",
            model_py="Model/squeezenet_msfu_ex1.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_z": 5, "tap_idx_y": 8, "pool_type": "guided"},
            notes="Equivalent configuration is already available in E1-3.",
        ),
    ],
    "E5": [
        ExperimentCase(
            key="E5I_fire3_pool5_fire9",
            title="Feature choice I: Fire3 + Pool5 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E5/E5I_Fire3_Pool5_Fire9",
            model_dir="Model/E5/E5I_Fire3_Pool5_Fire9",
            existing_result_dir="Result/E5/E5I_Fire3_Pool5_Fire9",
            model_py="Model/squeezenet_msfu_ex4.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_x": 12, "tap_idx_z": 4, "tap_idx_y": 8, "pool_type": "guided", "bqs_mode": "dual", "lambda_mode": "learnable", "lambda_fixed": 0.5},
        ),
        ExperimentCase(
            key="E5I_fire4_pool5_fire9",
            title="Feature choice I: Fire4 + Pool5 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E5/E5I_Fire4_Pool5_Fire9",
            model_dir="Model/E5/E5I_Fire4_Pool5_Fire9",
            existing_result_dir="Result/E5/E5I_Fire4_Pool5_Fire9",
            model_py="Model/squeezenet_msfu_ex4.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_x": 12, "tap_idx_z": 6, "tap_idx_y": 8, "pool_type": "guided", "bqs_mode": "dual", "lambda_mode": "learnable", "lambda_fixed": 0.5},
        ),
        ExperimentCase(
            key="E5_standard_pool3_pool5_fire9",
            title="Feature choice standard: Pool3 + Pool5 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E5/E5I_Pool3_Pool5_Fire9",
            model_dir="Model/E5/E5I_Pool3_Pool5_Fire9",
            existing_result_dir="Result/E1-3/MSFU_Full_BQS_Guided",
            model_py="Model/squeezenet_msfu_ex1.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_z": 5, "tap_idx_y": 8, "pool_type": "guided"},
            notes="Equivalent configuration is already available in E1-3.",
        ),
        ExperimentCase(
            key="E5II_pool3_fire5_fire9",
            title="Feature choice II: Pool3 + Fire5 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E5/E5II_Pool3_Fire5_Fire9",
            model_dir="Model/E5/E5II_Pool3_Fire5_Fire9",
            existing_result_dir="Result/E5/E5II_Pool3_Fire5_Fire9",
            model_py="Model/squeezenet_msfu_ex4.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_x": 12, "tap_idx_z": 5, "tap_idx_y": 7, "pool_type": "guided", "bqs_mode": "dual", "lambda_mode": "learnable", "lambda_fixed": 0.5},
        ),
        ExperimentCase(
            key="E5II_pool3_fire6_fire9",
            title="Feature choice II: Pool3 + Fire6 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E5/E5II_Pool3_Fire6_Fire9",
            model_dir="Model/E5/E5II_Pool3_Fire6_Fire9",
            existing_result_dir="Result/E5/E5II_Pool3_Fire6_Fire9",
            model_py="Model/squeezenet_msfu_ex4.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_x": 12, "tap_idx_z": 5, "tap_idx_y": 9, "pool_type": "guided", "bqs_mode": "dual", "lambda_mode": "learnable", "lambda_fixed": 0.5},
        ),
        ExperimentCase(
            key="E5II_pool3_fire7_fire9",
            title="Feature choice II: Pool3 + Fire7 + Fire9",
            trainer=GENERIC_TRAINER,
            result_dir="Result/E5/E5II_Pool3_Fire7_Fire9",
            model_dir="Model/E5/E5II_Pool3_Fire7_Fire9",
            existing_result_dir="Result/E5/E5II_Pool3_Fire7_Fire9",
            model_py="Model/squeezenet_msfu_ex4.py",
            model_kwargs={**LEGACY_MSFU_DEFAULTS, "exp_variant": "msfu", "msfu_mode": "full", "tap_idx_x": 12, "tap_idx_z": 5, "tap_idx_y": 10, "pool_type": "guided", "bqs_mode": "dual", "lambda_mode": "learnable", "lambda_fixed": 0.5},
        ),
    ],
    "E6": [
        ExperimentCase(
            key="E6_cnn_models_efficiency",
            title="Efficiency benchmark for SqNet, MSANet, and MSFUNet",
            trainer="MSFUNet_experiments/code/trainers/ex6_efficiency/benchmark_cnn_models_efficiency.py",
            result_dir="Result/E6_cnn_models",
            model_dir="Model/E6_cnn_models",
            default_data_root="Dataset/full",
            data_layout="imagefolder",
            metrics=("Parameters", "FLOPs", "Latency", "FPS", "Model size"),
            notes="Run once per --model: squeezenet, msanet35, msanet53, msfu.",
        ),
        ExperimentCase(
            key="E6_vit_tiny_efficiency",
            title="Efficiency benchmark for ViT-Tiny",
            trainer="MSFUNet_experiments/code/trainers/ex6_efficiency/benchmark_vit_tiny_efficiency.py",
            result_dir="Result/E6_vit_tiny",
            model_dir="Model/E6_vit_tiny",
            default_data_root="Dataset/full",
            data_layout="imagefolder",
            metrics=("Parameters", "Latency", "FPS", "Model size"),
        ),
        ExperimentCase(
            key="E6_vit_base_efficiency",
            title="Efficiency benchmark for ViT-B",
            trainer="MSFUNet_experiments/code/trainers/ex6_efficiency/benchmark_vit_base_efficiency.py",
            result_dir="Result/E6_vit_base",
            model_dir="Model/E6_vit_base",
            default_data_root="Dataset/full",
            data_layout="imagefolder",
            metrics=("Parameters", "Latency", "FPS", "Model size"),
        ),
        ExperimentCase(
            key="E6_resnet18_flops",
            title="FLOPs/size benchmark for ResNet-18",
            trainer="MSFUNet_experiments/code/trainers/ex6_efficiency/measure_resnet18_efficiency.py",
            result_dir="Result/E6_resnet18",
            model_dir="Model/E6_resnet18",
            default_data_root="Dataset/full",
            data_layout="none",
            metrics=("Parameters", "FLOPs", "Latency", "FPS", "Model size"),
        ),
    ],
}


def all_cases() -> List[ExperimentCase]:
    return [case for cases in EXPERIMENTS.values() for case in cases]


def find_case(key: str) -> ExperimentCase:
    for case in all_cases():
        if case.key == key:
            return case
    raise KeyError(f"Unknown experiment case: {key}")
