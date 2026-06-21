"""Static checks for metric filenames and schemas documented in the guide."""

from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


EXPECTED_SNIPPETS = {
    "code/trainers/msfunet_ablation/lopo_msfunet_ablation_trainer.py": [
        '["pig","acc","prec","rec","f1","spec","auc","macro_acc"]',
        '"summary.csv"',
    ],
    "code/trainers/ex2_backbone_comparison/lopo_squeezenet_baseline_trainer.py": [
        '["pig","thr","acc","precision","recall","f1","specificity","auc","params","latency_ms","fps"]',
        '"lopo_test_metrics.csv"',
    ],
    "code/trainers/ex2_backbone_comparison/lopo_resnet18_trainer.py": [
        '["pig","acc","prec","rec","f1","spec","auc","thr","params","lat_ms","fps"]',
        '"lopo_summary.csv"',
    ],
    "code/trainers/ex2_backbone_comparison/lopo_vit_tiny_trainer.py": [
        '["pig","thr","acc","precision","recall","f1","specificity","auc","params","latency_ms","fps"]',
        '"lopo_summary.csv"',
    ],
    "code/trainers/ex2_backbone_comparison/lopo_vit_base_trainer.py": [
        '["pig","acc","prec","rec","f1","spec","auc","thr","params","lat_ms","fps"]',
        '"lopo_summary.csv"',
    ],
    "code/trainers/ex2_backbone_comparison/lopo_msanet_trainer.py": [
        '["pig", "acc", "precision", "recall", "f1", "specificity", "auc", "thr", "params", "latency_ms", "fps"]',
        '"lopo_summary.csv"',
    ],
    "code/trainers/ex6_efficiency/benchmark_cnn_models_efficiency.py": [
        '["fold","params(M)","latency(ms/img)","fps"',
        'f"kfold_{args.model}_benchmark.csv"',
    ],
    "code/trainers/ex6_efficiency/benchmark_vit_tiny_efficiency.py": [
        '["fold","params(M)","latency(ms)","fps","acc","precision","recall","f1","auc"]',
        '"kfold_vit_benchmark.csv"',
    ],
    "code/trainers/ex6_efficiency/measure_resnet18_efficiency.py": [
        '["model", "params_m", "flops_g", "latency_ms", "fps", "size_mb"',
        '"resnet18_efficiency.csv"',
    ],
}


def main() -> None:
    for relative_path, snippets in EXPECTED_SNIPPETS.items():
        text = (PROJECT / relative_path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in text, (relative_path, snippet)
    print(f"[outputs] {len(EXPECTED_SNIPPETS)} trainer metric contracts verified")


if __name__ == "__main__":
    main()
