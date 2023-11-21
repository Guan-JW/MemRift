from importlib import import_module
from pathlib import Path
import torch

# 自动 import 已编译好的 extension
_ext = import_module("float_split._ext")

def split(t: torch.Tensor):
    return _ext.split(t)

def merge(exp, sm, dtype):
    return _ext.merge(exp, sm, dtype)