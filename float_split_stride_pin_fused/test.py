import torch, ctypes, os, time, random, numpy as np
from torch.utils.cpp_extension import load
import float_split_stride_pin_fused as fs


# ----------------------------------------------------------------------
# 3.  待测 tensor 生成器
def gen_inputs():
    B, C = 8192, 4096
    base = torch.randn(B, C, device='cuda')

    # ① contiguous
    yield "contig", base

    # ② 转置 => stride (1, 8)
    yield "transpose", base.t()

    # ③ 按列取 slice，带 storage_offset
    yield "slice-col", base[:, 100:2000]

    # ④ 高维 view & permute
    x4 = base.view(128, 64, 32, 128).permute(2, 0, 3, 1)  # 任意 stride
    yield "permute", x4

    # ⑤ 非对齐步长：每 2 行取 1 行
    yield "strided", base[::2, :]

    # ⑥ 切成 1-D view 再转回 (16,8) 并转置
    yield "reshape+T", base.reshape(-1).view(C, B).t()

# ----------------------------------------------------------------------
# 4.  主测试循环
def check(dtype, repeat=5):
    print(f"\n=== {dtype} ===")
    ok = 0
    stream = torch.cuda.Stream()

    for name, t in gen_inputs():
        t = t.to(dtype=dtype)
        st0 = (t.size(), t.stride(), t.storage_offset())

        pack_times = []
        unpack_times = []

        for _ in range(repeat):
            torch.cuda.synchronize()
            tic = time.time()
            exp, sm = fs.split(t, stream.cuda_stream)
            torch.cuda.synchronize()
            pack_times.append((time.time() - tic) * 1000)

            torch.cuda.synchronize()
            tic = time.time()
            out = fs.merge(exp, sm, t.shape, t.stride(), t.storage_offset(), t.dtype)
            torch.cuda.synchronize()
            unpack_times.append((time.time() - tic) * 1000)

        # 检查值与布局是否一致（只需要检查最后一次）
        same_val = torch.equal(out, t)
        same_layout = (out.size() == st0[0] and
                       out.stride() == st0[1] and
                       out.storage_offset() == st0[2])

        avg_pack = sum(pack_times) / repeat
        avg_unpack = sum(unpack_times) / repeat

        if same_val and same_layout:
            print(f"[✓] {name:<10}  pack={avg_pack:6.2f} ms  unpack={avg_unpack:6.2f} ms")
            ok += 1
        else:
            print(f"[✗] {name:<10}  value_eq={same_val}  layout_eq={same_layout}")
            print(f"{out=}")
            print(f"{t=}")
    return ok == 6


def main():
    torch.cuda.manual_seed(0)
    assert check(torch.float32)
    # 某些早期 GPU 驱动没有 bf16, 跳过即可
    if torch.cuda.is_bf16_supported():
        assert check(torch.bfloat16)
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()