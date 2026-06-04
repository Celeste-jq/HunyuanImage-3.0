from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunyuan_image_3.token_slice_utils import normalize_token_slices


def test_normalize_token_slices_converts_single_token_ids_to_unit_ranges():
    assert normalize_token_slices([33]) == [(33, 34)]


def test_normalize_token_slices_preserves_existing_ranges():
    assert normalize_token_slices([(33, 37)]) == [(33, 37)]
