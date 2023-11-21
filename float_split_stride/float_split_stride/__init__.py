from importlib import import_module
_ext = import_module("float_split_stride._ext")

def split(t):                             # 返回 (exp, sm)  uint8
    return _ext.split(t)

def merge(exp, sm, size, stride, offset, dtype):
    # prototype: 原张量，用来取 sizes/strides/offset/dtype
    return _ext.merge(
        exp, sm, size, stride, offset, dtype)
