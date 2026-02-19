#!/usr/bin/env python
"""Training script for curriculum learning.

DEPRECATED: Use `train-curriculum` command instead.
This wrapper is kept for backward compatibility.
"""

import sys
import warnings

warnings.warn(
    "Using scripts/train.py is deprecated. "
    "Please use the installed command: train-curriculum",
    DeprecationWarning,
    stacklevel=2,
)

from offline_robotic_manipulation_curriculum.scripts.train import main

if __name__ == "__main__":
    main()
