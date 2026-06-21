# -*- coding: utf-8 -*-
"""Command builder/runner for organized MSFUNet experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import os
import sys
import copy
from pathlib import Path
from typing import List

from experiments import EXPERIMENTS, GENERIC_TRAINER, UNIFIED_MODEL, ExperimentCase, all_cases, find_case
from reproducibility import print_audit, validate_paths


def _python_executable() -> str:
    """Prefer the project-local environment when this repository lives in one."""
    local_python = Path("bin/python")
    return str(local_python) if local_python.is_file() else sys.executable


def _quote_args(args: List[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _output_dirs(case: ExperimentCase, args: argparse.Namespace) -> tuple:
    """Keep current runs separate from the archived research outputs."""
    if getattr(args, "protocol", "paper_clean") == "paper_clean":
        return f"Result/runs/{case.key}", f"Model/checkpoints/{case.key}"
    return case.result_dir, case.model_dir


def build_command(case: ExperimentCase, args: argparse.Namespace) -> List[str]:
    trainer = case.trainer
    data_root = args.data_root or case.default_data_root
    result_dir, model_dir = _output_dirs(case, args)
    if trainer == GENERIC_TRAINER:
        model_source = UNIFIED_MODEL if getattr(args, "protocol", "paper_clean") == "paper_clean" else (case.model_py or UNIFIED_MODEL)
        command = [
            _python_executable(),
            trainer,
            "--data_root",
            data_root,
            "--roi_cfg",
            args.roi_cfg,
            "--result_root",
            "/".join(result_dir.split("/")[:-1]),
            "--model_root",
            "/".join(model_dir.split("/")[:-1]),
            "--exp_name",
            result_dir.split("/")[-1],
            "--model_py",
            model_source,
            "--model_fn",
            "build_model",
            "--model_kwargs",
            json.dumps(case.model_kwargs, separators=(",", ":")),
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
            "--nw",
            str(args.num_workers),
            "--img_size",
            str(args.img_size),
            "--tune_threshold",
            "--enable_temp_scaling",
        ]
        if getattr(args, "protocol", "paper_clean") == "legacy":
            command.extend([
                "--tta_hflip",
                "--use_class_weight",
                "--enable_thr_quantile_map",
                "--align_color_to_train",
                "--adabn",
            ])
        return command

    if trainer.endswith("lopo_squeezenet_baseline_trainer.py"):
        return [
            _python_executable(),
            trainer,
            "--path_d",
            data_root,
            "--path_r",
            result_dir,
            "--path_m",
            model_dir,
            "--roi_cfg",
            args.roi_cfg,
            "-e",
            str(args.epochs),
            "-b",
            str(args.batch),
            "--nw",
            str(args.num_workers),
            "--img_size",
            str(args.img_size),
            "--tune_threshold",
            "--temp_mode",
            "savebest",
        ]

    if trainer.endswith("lopo_resnet18_trainer.py") or trainer.endswith("lopo_vit_base_trainer.py"):
        return [
            _python_executable(),
            trainer,
            "--path_d",
            data_root,
            "--path_r",
            result_dir,
            "--path_m",
            model_dir,
            "--roi_cfg",
            args.roi_cfg,
            "-e",
            str(args.epochs),
            "-b",
            str(args.batch),
            "--nw",
            str(args.num_workers),
            "--img_size",
            str(args.img_size),
        ]

    if trainer.endswith("lopo_vit_tiny_trainer.py"):
        return [
            _python_executable(),
            trainer,
            "--data_dir",
            data_root,
            "--out_dir",
            result_dir,
            "--ckpt_dir",
            model_dir,
            "--roi_cfg",
            args.roi_cfg,
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
            "--nw",
            str(args.num_workers),
            "--img_size",
            str(args.img_size),
            "--timm_name",
            "vit_tiny_patch16_224",
        ]

    if trainer.endswith("lopo_msanet_trainer.py"):
        return [
            _python_executable(),
            trainer,
            "--path_d",
            data_root,
            "--path_r",
            result_dir,
            "--path_m",
            model_dir,
            "--roi_cfg",
            args.roi_cfg,
            "--model_type",
            "MSA_Addition_Pool35",
            "-e",
            str(args.epochs),
            "-b",
            str(args.batch),
            "--nw",
            str(args.num_workers),
            "--img_size",
            str(args.img_size),
        ]

    if trainer.endswith("benchmark_cnn_models_efficiency.py"):
        return [
            _python_executable(),
            trainer,
            "--data_dir",
            data_root,
            "--out_dir",
            result_dir,
            "--model",
            args.efficiency_model,
            "--epochs",
            str(args.efficiency_epochs),
            "--batch",
            str(args.batch),
            "--nw",
            str(args.num_workers),
            "--img_size",
            str(args.img_size),
        ]

    if trainer.endswith("benchmark_vit_tiny_efficiency.py"):
        return [
            _python_executable(),
            trainer,
            "--data_dir",
            data_root,
            "--out_dir",
            result_dir,
            "--timm_name",
            "vit_tiny_patch16_224",
            "--epochs",
            str(args.efficiency_epochs),
            "--batch",
            str(args.batch),
            "--nw",
            str(args.num_workers),
            "--img_size",
            str(args.img_size),
        ]

    if trainer.endswith("benchmark_vit_base_efficiency.py"):
        return [
            _python_executable(),
            trainer,
            "--data_dir",
            data_root,
            "--out_dir",
            result_dir,
            "--model_dir",
            model_dir,
            "-e",
            str(args.efficiency_epochs),
            "-b",
            str(args.batch),
            "--img_size",
            str(args.img_size),
            "--ckpt",
            "google/vit-base-patch16-224-in21k",
        ]

    if trainer.endswith("measure_resnet18_efficiency.py"):
        return [_python_executable(), trainer, "--out_dir", result_dir, "--img_size", str(args.img_size)]

    raise ValueError(f"No command builder for trainer: {trainer}")


def list_cases() -> None:
    for group, cases in EXPERIMENTS.items():
        print(f"\n[{group}]")
        for case in cases:
            existing = f" | existing={case.existing_result_dir}" if case.existing_result_dir else ""
            print(f"  {case.key}: {case.title}{existing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List, print, or run MSFUNet experiment cases.")
    parser.add_argument("--list", action="store_true", help="List experiment cases.")
    parser.add_argument("--audit", action="store_true", help="Check experiment configuration and available outputs without training.")
    parser.add_argument("--case", type=str, default="", help="Case key from --list.")
    parser.add_argument("--group", choices=sorted(EXPERIMENTS), help="Print or run every case in one paper experiment group.")
    parser.add_argument("--run", action="store_true", help="Execute the generated command.")
    parser.add_argument(
        "--protocol", choices=("legacy", "paper_clean"), default="paper_clean",
        help="legacy reproduces recorded runs; paper_clean disables TTA/class weighting/quantile mapping/ColorAlign/AdaBN.",
    )
    parser.add_argument("--data_root", type=str, default="", help="Override the case-specific dataset path.")
    parser.add_argument("--roi_cfg", type=str, default="Model/roi_pig.json")
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--efficiency_epochs", type=int, default=30)
    parser.add_argument("--efficiency_model", type=str, default="msfu", choices=["squeezenet", "msanet35", "msanet53", "msfu"])
    args = parser.parse_args()

    if args.case and args.group:
        parser.error("Use either --case or --group, not both.")

    if args.audit:
        print_audit()
        if not args.case and not args.group:
            return

    if args.list or (not args.case and not args.group):
        list_cases()
        if not args.case and not args.group:
            return

    jobs = []
    selected_cases = EXPERIMENTS[args.group] if args.group else [find_case(args.case)]
    for case in selected_cases:
        if args.group == "E6" and case.key == "E6_cnn_models_efficiency":
            for model_name in ("squeezenet", "msanet35", "msanet53", "msfu"):
                job_args = copy.copy(args)
                job_args.efficiency_model = model_name
                jobs.append((case, job_args))
        else:
            jobs.append((case, args))

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-msfunet")
    env["MSFUNET_DETERMINISTIC"] = "1" if args.protocol == "paper_clean" else "0"
    if args.protocol == "paper_clean":
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for case, job_args in jobs:
        data_root = job_args.data_root or case.default_data_root
        selected_model = UNIFIED_MODEL if (job_args.protocol == "paper_clean" and case.trainer == GENERIC_TRAINER) else (case.model_py or "")
        manifest = validate_paths(case, data_root, job_args.roi_cfg, selected_model)
        print(f"[preflight:{case.key}] " + json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        command = build_command(case, job_args)
        print(_quote_args(command))
        if args.run:
            subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
