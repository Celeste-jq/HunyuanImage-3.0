def unwrap_single_batch_value(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected a single-item list, got {len(value)} items")
        return value[0]

    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"Expected a single-item tuple, got {len(value)} items")
        return value[0]

    squeeze = getattr(value, "squeeze", None)
    if callable(squeeze):
        return squeeze(0)

    return value


def scalar_to_int(value) -> int:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)
