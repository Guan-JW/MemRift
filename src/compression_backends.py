"""Byte codecs used by the Table 6 activation-backend comparison."""

BACKENDS = ("lz4", "zstd", "ebc-lz4", "ebc-zstd")


def codec_name(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported compression backend: {backend}")
    return backend.rsplit("-", 1)[-1]


def uses_ebc(backend: str) -> bool:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported compression backend: {backend}")
    return backend.startswith("ebc-")


def validate_level(backend: str, level: int) -> None:
    codec = codec_name(backend)
    if codec == "zstd" and not -131072 <= level <= 22:
        raise ValueError("Zstd compression level must be between -131072 and 22")
    if codec == "lz4" and not 0 <= level <= 16:
        raise ValueError("LZ4 compression level must be between 0 and 16")


class ByteCodec:
    def __init__(self, backend: str, level: int) -> None:
        validate_level(backend, level)
        self.name = codec_name(backend)
        self.level = level
        if self.name == "zstd":
            import zstandard as zstd

            self.compressor = zstd.ZstdCompressor(level=level, write_checksum=False)
            self.decompressor = zstd.ZstdDecompressor()

    def compress(self, data) -> bytes:
        if self.name == "zstd":
            return self.compressor.compress(data)
        import lz4.frame

        return lz4.frame.compress(data, compression_level=self.level)

    def decompress(self, payload, expected_size: int) -> bytes:
        if self.name == "zstd":
            result = self.decompressor.decompress(payload, max_output_size=expected_size)
        else:
            import lz4.frame

            result = lz4.frame.decompress(payload)
        if len(result) != expected_size:
            raise ValueError(f"decompressed {len(result)} bytes; expected {expected_size}")
        return result


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float | None:
    return original_bytes / compressed_bytes if compressed_bytes else None
