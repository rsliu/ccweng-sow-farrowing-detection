# -*- coding: utf-8 -*-
"""Model factory used by experiment trainers."""

from __future__ import annotations

import pathlib
import sys

# This file is also loaded directly by importlib from trainer scripts.
# Add `code/` to sys.path so `models.*` imports work in that mode.
CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models._squeezenet_core import (  # noqa: E402
    MSFUConfig,
    ModelFactory,
    SQUEEZENET_TAPS,
    SqueezeNetConfig,
    build_model,
)

# Trainer scripts import this module by file path, so keep the public API small.
__all__ = [
    "MSFUConfig",
    "ModelFactory",
    "SQUEEZENET_TAPS",
    "SqueezeNetConfig",
    "build_model",
]
