from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunyuan_image_3.processor_output_utils import scalar_to_int, unwrap_single_batch_value


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def squeeze(self, dim):
        assert dim == 0
        return self.value


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def test_unwrap_single_batch_value_returns_first_item_for_singleton_list():
    value = ["pixel-values"]

    result = unwrap_single_batch_value(value)

    assert result == "pixel-values"


def test_unwrap_single_batch_value_squeezes_first_dim_for_tensor_batches():
    value = FakeTensor([["a", "b"], ["c", "d"]])

    result = unwrap_single_batch_value(value)

    assert result == [["a", "b"], ["c", "d"]]


def test_scalar_to_int_accepts_python_ints():
    assert scalar_to_int(12) == 12


def test_scalar_to_int_accepts_tensor_like_scalars():
    assert scalar_to_int(FakeScalar(34)) == 34
