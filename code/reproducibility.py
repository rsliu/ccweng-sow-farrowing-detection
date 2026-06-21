# -*- coding: utf-8 -*-
"""Dataset preflight checks and output summaries for the six experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from experiments import EXPERIMENTS, ExperimentCase


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _images(path: Path) -> List[Path]:
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def validate_dataset(case: ExperimentCase, data_root: str) -> Dict[str, object]:
    """Validate the data layout required by a case and return a compact manifest."""
    root = Path(data_root)
    if case.data_layout == "none":
        return {"layout": "none", "root": str(root), "images": 0}
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    class_dirs = [root / "exposed", root / "not_exposed"]
    missing = [str(p) for p in class_dirs if not p.is_dir()]
    if missing:
        raise ValueError(f"Expected binary class folders exposed/not_exposed; missing: {missing}")

    counts = {p.name: len(_images(p)) for p in class_dirs}
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"Both classes must contain images; counts={counts}")

    manifest: Dict[str, object] = {
        "layout": case.data_layout,
        "root": str(root.resolve()),
        "images": sum(counts.values()),
        "class_counts": counts,
    }
    if case.data_layout == "lopo":
        pigs_by_class = {
            class_dir.name: sorted(p.name for p in class_dir.iterdir() if p.is_dir() and _images(p))
            for class_dir in class_dirs
        }
        exposed = set(pigs_by_class["exposed"])
        not_exposed = set(pigs_by_class["not_exposed"])
        if not exposed or exposed != not_exposed:
            raise ValueError(
                "LOPO requires data_root/<class>/<pig_id>/<image>. "
                f"Pig folders differ by class: exposed={sorted(exposed)}, "
                f"not_exposed={sorted(not_exposed)}"
            )
        if len(exposed) < 3:
            raise ValueError(f"LOPO needs at least 3 pigs for train/val/test; found {sorted(exposed)}")
        manifest["pigs"] = sorted(exposed)
        manifest["pig_count"] = len(exposed)
        manifest["pig_class_counts"] = {
            f"{class_dir.name}/{pig}": len(_images(class_dir / pig))
            for class_dir in class_dirs for pig in sorted(exposed)
        }
    return manifest


def validate_paths(case: ExperimentCase, data_root: str, roi_cfg: str, model_py: str = "") -> Dict[str, object]:
    selected_model = model_py or case.model_py
    missing = [path for path in (case.trainer, selected_model) if path and not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Required source file(s) missing: {missing}")
    if case.data_layout == "lopo" and (not roi_cfg or not Path(roi_cfg).is_file()):
        raise FileNotFoundError(f"LOPO ROI configuration is missing: {roi_cfg}")
    return validate_dataset(case, data_root)


def _expected_run_outputs(case: ExperimentCase) -> List[Path]:
    root = Path("Result/runs") / case.key
    trainer = case.trainer
    if trainer == "code/trainers/msfunet_ablation/lopo_msfunet_ablation_trainer.py":
        return [root / "summary.csv"]
    if trainer.endswith("lopo_squeezenet_baseline_trainer.py"):
        return [root / "lopo_test_metrics.csv"]
    if "/ex2_backbone_comparison/" in trainer:
        return [root / "lopo_summary.csv"]
    if trainer.endswith("benchmark_cnn_models_efficiency.py"):
        return [root / f"kfold_{model}_benchmark.csv" for model in ("squeezenet", "msanet35", "msanet53", "msfu")]
    if trainer.endswith("benchmark_vit_tiny_efficiency.py"):
        return [root / "kfold_vit_benchmark.csv"]
    if trainer.endswith("benchmark_vit_base_efficiency.py"):
        return [root / "MHvit1_result.txt"]
    if trainer.endswith("measure_resnet18_efficiency.py"):
        return [root / "resnet18_efficiency.csv"]
    return []


def _valid_run_output(case: ExperimentCase, path: Path) -> bool:
    """Validate the minimum metric schema and fold coverage of a result file."""
    if not path.is_file():
        return False
    if path.suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        return "[MODEL STATS]" in text and "Params" in text and "latency" in text

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return False
    header = {name.strip().lower() for name in rows[0]}
    aliases = {
        "accuracy": {"acc", "accuracy", "acc(%)"},
        "precision": {"prec", "precision", "prec(%)"},
        "recall": {"rec", "recall", "rec(%)"},
        "f1": {"f1", "f1(%)"},
        "specificity": {"spec", "specificity", "spec(%)"},
        "auc": {"auc"},
        "params": {"params", "params(m)", "params_m"},
        "latency": {"lat_ms", "latency_ms", "latency(ms)", "latency(ms/img)"},
        "fps": {"fps"},
    }
    if "/ex6_efficiency/" in case.trainer:
        required = ("params", "latency", "fps")
        minimum_rows = 2
    elif "/ex2_backbone_comparison/" in case.trainer:
        required = ("accuracy", "precision", "recall", "f1", "specificity", "auc")
        minimum_rows = 9
    else:
        required = ("accuracy", "precision", "recall", "f1", "specificity", "auc")
        minimum_rows = 9
    return len(rows) >= minimum_rows and all(header & aliases[name] for name in required)


def audit_existing_results() -> List[Tuple[str, str, str, str, str]]:
    """Return configuration, archived-output, and current-run status."""
    rows: List[Tuple[str, str, str, str, str]] = []
    for group, cases in EXPERIMENTS.items():
        existing = [case for case in cases if case.existing_result_dir]
        missing_dirs = [case.key for case in existing if not Path(case.existing_result_dir).is_dir()]
        summaries = []
        for case in existing:
            root = Path(case.existing_result_dir)
            summaries.extend(root.glob("summary.csv"))
            summaries.extend(root.glob("lopo_summary.csv"))
        if missing_dirs:
            reference_status = "INCOMPLETE"
            detail = "missing reference directories: " + ", ".join(missing_dirs)
        elif group == "E6":
            outputs = list(Path("Result/E6_cnn_models").glob("*.csv")) if Path("Result/E6_cnn_models").is_dir() else []
            reference_status = "COMPLETE" if outputs else "NOT_RUN"
            detail = f"reference benchmark CSV files: {len(outputs)}"
        else:
            incomplete = 0
            for summary in set(summaries):
                with summary.open("r", encoding="utf-8", newline="") as f:
                    header = {name.strip().lower() for name in next(csv.reader(f), [])}
                aliases = {
                    "accuracy": {"acc", "accuracy", "acc(%)"},
                    "f1": {"f1", "f1(%)"},
                    "auc": {"auc"},
                }
                if any(not (names & header) for names in aliases.values()):
                    incomplete += 1
            reference_status = "COMPLETE" if incomplete == 0 else "PARTIAL"
            detail = f"registered cases={len(cases)}, summaries={len(set(summaries))}, incomplete metric schemas={incomplete}"
        expected_pairs = [(case, path) for case in cases for path in _expected_run_outputs(case)]
        present = sum(path.is_file() for _, path in expected_pairs)
        valid = sum(_valid_run_output(case, path) for case, path in expected_pairs)
        run_status = "COMPLETE" if expected_pairs and valid == len(expected_pairs) else ("PARTIAL" if present else "NOT_RUN")
        rows.append((group, "READY", reference_status, run_status, f"{detail}; run files={present}/{len(expected_pairs)}, valid={valid}/{len(expected_pairs)}"))
    return rows


def print_audit() -> None:
    print("experiment,configuration,archived_outputs,run_outputs,detail")
    for group, configuration, archived, run_outputs, detail in audit_existing_results():
        print(f"{group},{configuration},{archived},{run_outputs},{detail}")
