from pathlib import Path


MODELING_SOURCE = Path("hunyuan_image_3/modeling_hunyuan_image_3.py")
PIPELINE_SOURCE = Path("hunyuan_image_3/hunyuan_image_3_pipeline.py")
AUTOENCODER_SOURCE = Path("hunyuan_image_3/autoencoder_kl_3d.py")


def test_to_device_recurses_into_dict_values():
    source = MODELING_SOURCE.read_text()

    assert "elif isinstance(data, dict):" in source
    assert "return {key: to_device(value, device) for key, value in data.items()}" in source


def test_core_generation_paths_do_not_hardcode_cuda_autocast():
    modeling_source = MODELING_SOURCE.read_text()
    pipeline_source = PIPELINE_SOURCE.read_text()

    assert 'torch.autocast(device_type="cuda"' not in modeling_source
    assert 'torch.autocast(device_type="cuda"' not in pipeline_source
    assert "device_autocast(" in modeling_source
    assert "device_autocast(" in pipeline_source


def test_prepare_model_inputs_has_npu_diagnostics():
    source = MODELING_SOURCE.read_text()

    assert "debug_npu = kwargs.get(\"debug_npu\", False)" in source
    assert "self._print_npu_debug_info(" in source
    assert "cond_vit_image_kwargs" in source


def test_vae_runtime_paths_do_not_allocate_cuda_directly():
    source = AUTOENCODER_SOURCE.read_text()

    assert 'self.empty_cache = torch.empty(0, device="cuda")' not in source
    assert "z = z.cuda()" not in source
