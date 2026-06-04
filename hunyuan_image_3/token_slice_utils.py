from typing import Iterable


def normalize_token_slices(values) -> list[tuple[int, int]]:
    if values is None:
        return []

    normalized: list[tuple[int, int]] = []
    for value in values:
        if isinstance(value, int):
            normalized.append((value, value + 1))
            continue

        if isinstance(value, tuple) and len(value) == 2:
            normalized.append((int(value[0]), int(value[1])))
            continue

        if isinstance(value, list) and len(value) == 2:
            normalized.append((int(value[0]), int(value[1])))
            continue

        raise TypeError(f"Unsupported token slice value: {value!r}")

    return normalized
