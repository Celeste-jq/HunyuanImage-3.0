from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_image_gen_npu import build_model_kwargs, parse_args, parse_image_input


def test_parse_image_input_handles_csv_paths():
    result = parse_image_input(" a.png, b.png ,, c.png ")

    assert result == ["a.png", "b.png", "c.png"]


def test_build_model_kwargs_prefers_npu_safe_defaults():
    args = SimpleNamespace(
        attn_impl="sdpa",
        torch_dtype="auto",
        device_map="auto",
        moe_impl="eager",
    )

    assert build_model_kwargs(args) == {
        "attn_implementation": "sdpa",
        "torch_dtype": "auto",
        "device_map": "auto",
        "moe_impl": "eager",
        "moe_drop_tokens": True,
    }


def test_npu_entrypoint_defaults_to_non_streaming_verbose_level(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_image_gen_npu.py", "--prompt", "p"])

    args = parse_args()

    assert args.verbose == 1


def test_npu_debug_flag_defaults_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_image_gen_npu.py", "--prompt", "p"])

    args = parse_args()

    assert args.debug_npu is False


def test_npu_debug_flag_can_be_enabled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_image_gen_npu.py", "--prompt", "p", "--debug-npu"])

    args = parse_args()

    assert args.debug_npu is True
