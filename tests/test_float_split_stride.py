import importlib

import pytest


torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)
try:
    split_stride = importlib.import_module("float_split_stride")
except (ImportError, OSError) as error:
    pytest.skip(f"float_split_stride extension is unavailable: {error}", allow_module_level=True)


def roundtrip(source):
    stream = torch.cuda.current_stream(device=source.device)
    exponent, sign_mantissa = split_stride.split(source, stream.cuda_stream)
    restored = split_stride.merge(
        exponent,
        sign_mantissa,
        source.shape,
        source.stride(),
        source.storage_offset(),
        source.dtype,
        stream.cuda_stream,
    )
    stream.synchronize()
    return restored


def assert_bit_exact(actual, expected):
    integer_dtype = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.contiguous().view(integer_dtype), expected.contiguous().view(integer_dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(3,), (3, 5), (1, 3, 5), (513,)])
def test_required_odd_contiguous_shapes_are_bit_exact(dtype, shape):
    source = torch.randn(shape, device="cuda", dtype=dtype)
    assert_bit_exact(roundtrip(source), source)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_noncontiguous_tensor_is_bit_exact(dtype):
    source = torch.randn((3, 10), device="cuda", dtype=dtype)[:, 1::2]
    assert not source.is_contiguous()
    restored = roundtrip(source)
    assert restored.stride() == source.stride()
    assert_bit_exact(restored, source)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_storage_offset_tensor_is_bit_exact(dtype):
    source = torch.randn(515, device="cuda", dtype=dtype)[1:514]
    assert source.storage_offset() == 1
    restored = roundtrip(source)
    assert restored.storage_offset() == source.storage_offset()
    assert_bit_exact(restored, source)
