# -*- coding: utf-8 -*-
"""Metric helpers used by training and benchmark scripts."""

from __future__ import annotations

import numpy as np


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator) / float(denominator) if denominator != 0 else float(default)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            matrix[target, pred] += 1
    return matrix


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        pos = y_true == 1
        neg = y_true == 0
        num_pos = int(pos.sum())
        num_neg = int(neg.sum())
        if num_pos == 0 or num_neg == 0:
            return float("nan")

        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
        sum_ranks_pos = ranks[pos].sum()
        auc = (sum_ranks_pos - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
        return float(auc)
