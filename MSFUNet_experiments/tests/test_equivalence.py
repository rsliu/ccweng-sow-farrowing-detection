"""Numerical equivalence checks for the organized MSFUNet implementation."""

from __future__ import annotations

import collections
import importlib.util
from pathlib import Path
import sys
import warnings

import torch


PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "MSFUNet_experiments" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


CASES = (
    ("Model/squeezenet_msfu_ex1.py", {"exp_variant": "msfu", "msfu_mode": "full", "tap_idx_z": 5, "tap_idx_y": 8, "pool_type": "guided"}),
    ("Model/squeezenet_msfu_ex2.py", {"exp_variant": "msfu", "msfu_mode": "full", "tap_idx_z": 5, "tap_idx_y": 8, "pool_type": "guided", "bqs_mode": "shal"}),
    ("Model/squeezenet_msfu_ex4.py", {"exp_variant": "msfu", "msfu_mode": "full", "tap_idx_x": 12, "tap_idx_z": 5, "tap_idx_y": -1, "pool_type": "guided"}),
)


def _load_builder(path: Path):
    spec = importlib.util.spec_from_file_location(f"reference_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_model


def main() -> None:
    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
    from models.factory import build_model
    from models.comparison_models import SqueezeNetSimple
    from models._squeezenet_core import SqueezeNetWithAttention

    checked = 0
    for relative_path, kwargs in CASES:
        reference_path = PROJECT / relative_path
        if not reference_path.is_file():
            continue
        reference = _load_builder(reference_path)(num_classes=2, **kwargs).eval()
        organized = build_model(num_classes=2, **kwargs).eval()
        reference_state = list(reference.state_dict().values())
        organized_state = organized.state_dict()
        assert [tuple(value.shape) for value in reference_state] == [tuple(value.shape) for value in organized_state.values()]
        mapped = collections.OrderedDict(
            (key, value.detach().clone())
            for (key, _), value in zip(organized_state.items(), reference_state)
        )
        organized.load_state_dict(mapped, strict=True)
        sample = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            difference = (reference(sample) - organized(sample)).abs().max().item()
        assert difference < 1e-7, (relative_path, difference)
        checked += 1

    comparison_cases = (
        ("Model/squeezenet_without_MSA.py", "SqueezeNetSimple", SqueezeNetSimple, {}),
        ("Model/squeezenet_multi_scale_attention_zy_addition.py", "SqueezeNetWithAttention", SqueezeNetWithAttention, {"order": "pool3_pool5"}),
        ("Model/squeezenet_multi_scale_attention_yz_addition.py", "SqueezeNetWithAttention", SqueezeNetWithAttention, {"order": "pool5_pool3"}),
    )
    for relative_path, class_name, organized_class, organized_kwargs in comparison_cases:
        reference_path = PROJECT / relative_path
        if not reference_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(f"reference_{reference_path.stem}", reference_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        reference = getattr(module, class_name)(num_classes=2).eval()
        organized = organized_class(num_classes=2, **organized_kwargs).eval()
        organized_state = organized.state_dict()
        if class_name == "SqueezeNetWithAttention":
            replacements = (
                ("squeezenet.", "backbone."),
                ("attention.gamma", "gamma"),
                ("attention.f_conv", "query"),
                ("attention.g_conv", "key"),
                ("attention.h_conv", "low_proj"),
                ("attention.u_conv", "mid_proj"),
                ("attention.channel_conv", "channel_proj"),
            )
            mapped = collections.OrderedDict()
            for key, value in reference.state_dict().items():
                target = key
                for old, new in replacements:
                    if target.startswith(old):
                        target = new + target[len(old):]
                        break
                mapped[target] = value.detach().clone()
        else:
            reference_state = list(reference.state_dict().values())
            assert [tuple(value.shape) for value in reference_state] == [tuple(value.shape) for value in organized_state.values()]
            mapped = collections.OrderedDict(
                (key, value.detach().clone()) for (key, _), value in zip(organized_state.items(), reference_state)
            )
        organized.load_state_dict(mapped, strict=True)
        sample = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            difference = (reference(sample) - organized(sample)).abs().max().item()
        assert difference < 1e-7, (relative_path, difference)
        checked += 1
    assert checked in (0, len(CASES) + len(comparison_cases))
    print(f"[equivalence] {checked} reference variants numerically matched")


if __name__ == "__main__":
    main()
