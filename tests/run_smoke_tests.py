# -*- coding: utf-8 -*-
"""Dependency-free smoke test runner.

Use this when pytest is not installed:

    python tests/run_smoke_tests.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from experiments import EXPERIMENTS, all_cases, find_case  # noqa: E402
from models.factory import build_model  # noqa: E402
from run import build_command  # noqa: E402
from test_utils import main as utils_ok  # noqa: E402
from test_output_contracts import main as outputs_ok  # noqa: E402
from test_reproducibility import main as data_contract_ok  # noqa: E402


def forward_ok(name: str, **kwargs) -> None:
    print(f"[model] {name}")
    model = build_model(num_classes=2, pretrained=False, qk_dim=16, **kwargs)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 2), (name, tuple(out.shape))


def registry_ok() -> None:
    print("[registry] six experiment groups")
    assert set(EXPERIMENTS) == {"E1", "E2", "E3", "E4", "E5", "E6"}

    args = argparse.Namespace(
        data_root="",
        roi_cfg="config/roi_pig.json",
        epochs=1,
        batch=2,
        num_workers=0,
        img_size=224,
        efficiency_epochs=1,
        efficiency_model="msfu",
        protocol="paper_clean",
    )
    case = find_case("E1_msfunet_full")
    command = build_command(case, args)
    assert "--model_py" in command
    assert "Dataset/lopo" in command
    for registered_case in all_cases():
        generated = build_command(registered_case, args)
        assert generated[0].endswith("python")
        assert generated[1] == registered_case.trainer
    clean_command = build_command(case, args)
    assert "Result/runs" in clean_command
    assert "Model/checkpoints" in clean_command
    assert "E1_msfunet_full" in clean_command
    assert "code/models/factory.py" in clean_command
    for flag in ("--tta_hflip", "--use_class_weight", "--enable_thr_quantile_map", "--align_color_to_train", "--adabn"):
        assert flag not in clean_command


def registered_models_ok() -> None:
    """Run every unique registered MSFUNet configuration through the public factory."""
    seen = set()
    checked = 0
    for case in all_cases():
        if not case.model_kwargs:
            continue
        signature = repr(sorted(case.model_kwargs.items()))
        if signature in seen:
            continue
        seen.add(signature)
        kwargs = dict(case.model_kwargs)
        kwargs["pretrained"] = False
        model = build_model(num_classes=2, **kwargs).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        assert tuple(output.shape) == (1, 2), case.key
        checked += 1
    assert checked >= 10
    print(f"[registry-models] {checked} unique registered configurations")


def main() -> None:
    forward_ok("baseline", variant="baseline", pool_type="gap")
    forward_ok("fusion_only", variant="fusion_only", msfu_mode="fusion_only", pool_type="gap")
    forward_ok("msfunet_full", variant="msfunet", msfu_mode="full", pool_type="guided")
    forward_ok("deep_only", variant="msfunet", msfu_mode="full", bqs_mode="deep", pool_type="guided")
    forward_ok("disable_mid", variant="msfunet", msfu_mode="full", tap_idx_z=5, tap_idx_y=-1, pool_type="guided")
    forward_ok("msanet35", variant="msanet35")
    utils_ok()
    registry_ok()
    registered_models_ok()
    outputs_ok()
    data_contract_ok()
    print("[ok] smoke tests passed")


if __name__ == "__main__":
    main()
