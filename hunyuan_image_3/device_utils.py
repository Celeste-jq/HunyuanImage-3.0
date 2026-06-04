from contextlib import contextmanager

import torch


def device_type_of(value, fallback: str = "cpu") -> str:
    if isinstance(value, torch.Tensor):
        return value.device.type
    if isinstance(value, torch.device):
        return value.type
    if isinstance(value, str):
        return value.split(":", 1)[0]
    if isinstance(value, dict):
        for item in value.values():
            return device_type_of(item, fallback=fallback)
    if isinstance(value, (list, tuple)):
        for item in value:
            return device_type_of(item, fallback=fallback)
    return fallback


@contextmanager
def device_autocast(device_like, dtype, enabled: bool = True):
    if not enabled or dtype == torch.float32:
        yield
        return

    device_type = device_type_of(device_like)
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
        yield
