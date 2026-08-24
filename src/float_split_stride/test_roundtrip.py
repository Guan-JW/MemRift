import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


def assert_bit_exact(testcase, actual, expected):
    integer_dtype = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    testcase.assertTrue(torch.equal(actual.contiguous().view(integer_dtype), expected.contiguous().view(integer_dtype)))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
class RoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import float_split_stride
        cls.split_stride = float_split_stride

    def roundtrip(self, source):
        stream = torch.cuda.current_stream()
        exponent, sign_mantissa = self.split_stride.split(source, stream.cuda_stream)
        restored = self.split_stride.merge(
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

    def test_odd_contiguous_roundtrip(self):
        for dtype in (torch.bfloat16, torch.float32):
            for shape in ((3,), (3, 5), (1, 3, 5), (513,)):
                with self.subTest(dtype=dtype, shape=shape):
                    source = torch.randn(shape, device="cuda", dtype=dtype)
                    assert_bit_exact(self, self.roundtrip(source), source)

    def test_noncontiguous_roundtrip(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                source = torch.randn((3, 10), device="cuda", dtype=dtype)[:, 1::2]
                self.assertFalse(source.is_contiguous())
                restored = self.roundtrip(source)
                self.assertEqual(restored.stride(), source.stride())
                assert_bit_exact(self, restored, source)

    def test_storage_offset_roundtrip(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                source = torch.randn(515, device="cuda", dtype=dtype)[1:514]
                self.assertEqual(source.storage_offset(), 1)
                restored = self.roundtrip(source)
                self.assertEqual(restored.storage_offset(), source.storage_offset())
                assert_bit_exact(self, restored, source)

    def test_empty_roundtrip(self):
        for dtype in (torch.bfloat16, torch.float32):
            for shape in ((0,), (2, 0, 3)):
                with self.subTest(dtype=dtype, shape=shape):
                    source = torch.empty(shape, device="cuda", dtype=dtype)
                    restored = self.roundtrip(source)
                    self.assertEqual(restored.shape, source.shape)
                    self.assertEqual(restored.numel(), 0)

    def test_more_than_four_dimensions_is_rejected(self):
        source = torch.empty((1, 1, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.current_stream()
        with self.assertRaisesRegex(RuntimeError, "at most 4 dimensions"):
            self.split_stride.split(source, stream.cuda_stream)


if __name__ == "__main__":
    unittest.main()
