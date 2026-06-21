# -*- coding: utf-8 -*-
"""Compare registered E1/E3/E4/E5 settings with recorded run configs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "code"))

from experiments import GENERIC_TRAINER, all_cases  # noqa: E402
from run import build_command  # noqa: E402


def _cli_args():
    return argparse.Namespace(
        data_root="", roi_cfg="Model/roi_pig.json", epochs=55, batch=16,
        num_workers=4, img_size=224, efficiency_epochs=30, efficiency_model="msfu",
        protocol="legacy",
    )


def main() -> None:
    trainer_path = PROJECT / GENERIC_TRAINER
    spec = importlib.util.spec_from_file_location("registered_generic_trainer", trainer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.build_argparser()

    checked = 0
    for case in all_cases():
        if case.trainer != GENERIC_TRAINER or case.existing_result_dir != case.result_dir:
            continue
        config_path = PROJECT / case.existing_result_dir / "config.json"
        if not config_path.is_file():
            continue
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
        command = build_command(case, _cli_args())
        generated = vars(parser.parse_args(command[2:]))

        # Paths may be absolute in historical configs. Output, model, and ROI
        # locations must retain their expected project-relative suffixes.
        assert pathlib.Path(generated["data_root"]).as_posix().endswith("Dataset/lopo"), case.key
        for key in ("result_root", "model_root", "model_py", "roi_cfg"):
            assert pathlib.Path(generated[key]).name == pathlib.Path(recorded[key]).name, (case.key, key)
        assert json.loads(generated["model_kwargs"]) == json.loads(recorded["model_kwargs"]), case.key

        ignored = {"data_root", "result_root", "model_root", "model_py", "model_kwargs", "roi_cfg"}
        for key, value in generated.items():
            if key not in ignored and key in recorded:
                assert recorded[key] == value, (case.key, key, recorded[key], value)
        checked += 1

    assert checked == 14, checked
    print(f"[recipes] {checked} recorded MSFUNet recipes match registered commands")


if __name__ == "__main__":
    main()
