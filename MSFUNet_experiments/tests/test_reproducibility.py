"""Dataset-contract checks for the two experiment protocols."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "MSFUNet_experiments" / "code"
sys.path.insert(0, str(CODE))

from experiments import find_case  # noqa: E402
from reproducibility import validate_dataset  # noqa: E402


def main() -> None:
    lopo_root = PROJECT / "Dataset" / "lopo"
    flat_root = PROJECT / "Dataset" / "full"
    if lopo_root.is_dir():
        manifest = validate_dataset(find_case("E1_msfunet_full"), str(lopo_root))
        assert manifest["pig_count"] == 8
        assert manifest["images"] == 48000
    if flat_root.is_dir():
        manifest = validate_dataset(find_case("E6_cnn_models_efficiency"), str(flat_root))
        assert manifest["images"] > 0
        try:
            validate_dataset(find_case("E1_msfunet_full"), str(flat_root))
        except ValueError:
            pass
        else:
            raise AssertionError("The flat E6 dataset was incorrectly accepted as LOPO data")
    print("[datasets] expected dataset layouts are valid")


if __name__ == "__main__":
    main()
