from pathlib import Path


def test_hunyuan_static_cache_initializes_static_layers_with_key_and_value_states():
    source = Path("hunyuan_image_3/modeling_hunyuan_image_3.py").read_text()

    assert "lazy_initialization(key_states, value_states)" in source


def test_moe_forward_does_not_require_cuda_nvtx():
    source = Path("hunyuan_image_3/modeling_hunyuan_image_3.py").read_text()

    assert "with optional_nvtx_range(\"MoE\"):" in source
    assert "if hidden_states.device.type == \"cuda\":" in source
    assert "torch.cuda.set_device(hidden_states.device.index)" in source


def test_generation_kwargs_preserve_use_cache_for_transformers_sample_loop():
    source = Path("hunyuan_image_3/modeling_hunyuan_image_3.py").read_text()

    assert "                \"use_cache\": kwargs.get(\"use_cache\")," in source
    assert "updated_model_kwargs[\"use_cache\"] = model_kwargs.get(\"use_cache\")" in source
