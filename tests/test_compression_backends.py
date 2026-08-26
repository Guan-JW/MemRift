import importlib.util

import pytest

from conftest import load_module


codec = load_module("src/compression_backends.py", "compression_backends_test")


@pytest.mark.parametrize("backend", codec.BACKENDS)
def test_backend_byte_roundtrip(backend):
    dependency = "zstandard" if codec.codec_name(backend) == "zstd" else "lz4"
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is installed in the artifact image")
    payload = bytes(range(256)) * 17
    instance = codec.ByteCodec(backend, 1)
    compressed = instance.compress(payload)
    assert instance.decompress(compressed, len(payload)) == payload


def test_ebc_and_full_tensor_classification():
    assert codec.uses_ebc("ebc-zstd")
    assert codec.uses_ebc("ebc-lz4")
    assert not codec.uses_ebc("zstd")
    assert not codec.uses_ebc("lz4")


def test_backend_level_validation():
    with pytest.raises(ValueError, match="LZ4"):
        codec.ByteCodec("lz4", -1)
    with pytest.raises(ValueError, match="Zstd"):
        codec.ByteCodec("zstd", 23)


def test_compression_ratio_accounts_for_complete_payload():
    assert codec.compression_ratio(200, 125) == 1.6
    assert codec.compression_ratio(200, 0) is None
