# -*- coding: utf-8 -*-
"""Smoke tests for experiment registry and command construction."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from experiments import EXPERIMENTS, all_cases, find_case  # noqa: E402
from run import build_command  # noqa: E402


def _args():
    return argparse.Namespace(
        data_root="",
        roi_cfg="Model/roi_pig.json",
        epochs=1,
        batch=2,
        num_workers=0,
        img_size=224,
        efficiency_epochs=1,
        efficiency_model="msfu",
        protocol="legacy",
    )


def test_registry_has_six_experiments():
    assert set(EXPERIMENTS) == {"E1", "E2", "E3", "E4", "E5", "E6"}
    assert len(all_cases()) >= 6


def test_find_case_and_build_generic_command():
    case = find_case("E1_msfunet_full")
    command = build_command(case, _args())
    assert command[0].endswith("python")
    assert "--model_py" in command
    assert "Model/squeezenet_msfu_ex1.py" in command
    assert "Dataset/lopo" in command
    for flag in ("--tta_hflip", "--use_class_weight", "--enable_thr_quantile_map", "--align_color_to_train", "--adabn"):
        assert flag in command


def test_build_backbone_command():
    case = find_case("E2_resnet18_lopo")
    command = build_command(case, _args())
    assert "MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_resnet18_trainer.py" in command
    assert "--path_d" in command


def test_efficiency_uses_flat_dataset_and_explicit_outputs():
    case = find_case("E6_vit_base_efficiency")
    command = build_command(case, _args())
    assert "Dataset/full" in command
    assert "--out_dir" in command
    assert "--model_dir" in command


def test_paper_clean_disables_auxiliary_options():
    args = _args()
    args.protocol = "paper_clean"
    command = build_command(find_case("E1_msfunet_full"), args)
    assert "Result/runs" in command
    assert "Model/checkpoints" in command
    assert "E1_msfunet_full" in command
    assert "MSFUNet_experiments/code/models/factory.py" in command
    for flag in ("--tta_hflip", "--use_class_weight", "--enable_thr_quantile_map", "--align_color_to_train", "--adabn"):
        assert flag not in command
