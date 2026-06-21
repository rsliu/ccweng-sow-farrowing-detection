# -*- coding: utf-8 -*-
"""Entry point for listing and running MSFUNet experiment cases."""

from __future__ import annotations

import pathlib
import sys


CODE_DIR = pathlib.Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE_DIR))

from run import main  # noqa: E402


if __name__ == "__main__":
    main()
