"""Check source references from organized trainers to the research files."""

from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PAIRS = {
    "train_lopo_generic.py": "MSFUNet_experiments/code/trainers/msfunet_ablation/lopo_msfunet_ablation_trainer.py",
    "train_lopo_msanet.py": "MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_msanet_trainer.py",
    "resNet.py": "MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_resnet18_trainer.py",
    "train_without_msa_v2.py": "MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_squeezenet_baseline_trainer.py",
    "train_vit.py": "MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_vit_base_trainer.py",
    "train_vit_small.py": "MSFUNet_experiments/code/trainers/ex2_backbone_comparison/lopo_vit_tiny_trainer.py",
    "train_mo_1pig_sque_fps.py": "MSFUNet_experiments/code/trainers/ex6_efficiency/benchmark_cnn_models_efficiency.py",
    "train_mo_1pig_vit_timm_fps.py": "MSFUNet_experiments/code/trainers/ex6_efficiency/benchmark_vit_tiny_efficiency.py",
}


def main() -> None:
    for original, organized in PAIRS.items():
        original_path = PROJECT / original
        assert original_path.is_file(), original
        organized_text = (PROJECT / organized).read_text(encoding="utf-8")
        assert f"Original source: {original}" in organized_text[:500], (original, organized)
    print(f"[sources] {len(PAIRS)} organized trainers include valid source declarations")


if __name__ == "__main__":
    main()
