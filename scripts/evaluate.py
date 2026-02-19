#!/usr/bin/env python
"""Evaluation script for trained policies.

DEPRECATED: Use `evaluate-curriculum` command instead.
This wrapper is kept for backward compatibility.
"""

import sys
import warnings

warnings.warn(
    "Using scripts/evaluate.py is deprecated. "
    "Please use the installed command: evaluate-curriculum",
    DeprecationWarning,
    stacklevel=2,
)

from offline_robotic_manipulation_curriculum.scripts.evaluate import main

if __name__ == "__main__":
    main()
