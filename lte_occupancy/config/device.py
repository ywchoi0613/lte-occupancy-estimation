"""
config/device.py — Execution-environment settings (GPU / compute device).

Deliberately separated from the research configuration: device selection depends
on the machine, not the experiment, so it is driven by environment variables
(``CUDA_VISIBLE_DEVICES``, ``LTE_GPU_INDEX``, ``XGB_FORCE_CPU``) rather than the
config object. Research parameters live in ``config/defaults.py`` only.
"""
from __future__ import annotations

import os

import torch

GPU_INDEX = int(os.environ.get("LTE_GPU_INDEX", 0))
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    _n = torch.cuda.device_count()
    if GPU_INDEX >= _n:
        print(f"WARN requested cuda:{GPU_INDEX} but only {_n} GPU(s) visible; using cuda:0")
        GPU_INDEX = 0
    DEVICE = torch.device(f"cuda:{GPU_INDEX}")
    print(f"CUDA available: True  | GPUs: {_n}  | using {DEVICE} "
          f"({torch.cuda.get_device_name(GPU_INDEX)})")
else:
    DEVICE = torch.device("cpu")
    print("CUDA available: False | using CPU")

_XGB_FORCE_CPU = bool(int(os.environ.get("XGB_FORCE_CPU", "0")))
XGB_DEVICE = "cpu" if (_XGB_FORCE_CPU or not USE_CUDA) else f"cuda:{GPU_INDEX}"
print(f"XGBoost device: {XGB_DEVICE}")
