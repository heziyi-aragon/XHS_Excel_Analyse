"""Copy-on-write patches shared by the portfolio tool and graph adapter."""
from copy import deepcopy


def merge_config(base: dict, patch: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(patch, dict):
        raise ValueError('Portfolio configuration and patches must be mappings')
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def changed_patch(current: dict, previous: dict) -> dict:
    """Omitted keys preserve old settings; an explicit feedback patch can reset any key."""
    if not isinstance(current, dict) or not isinstance(previous, dict):
        raise ValueError('Portfolio configuration must be a mapping')
    result = {}
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(previous.get(key), dict):
            nested = changed_patch(value, previous[key])
            if nested:
                result[key] = nested
        elif key not in previous or value != previous[key]:
            result[key] = deepcopy(value)
    return result


def paths_of(patch: dict, prefix: str = '') -> list[str]:
    result = []
    for key, value in patch.items():
        path = prefix + str(key)
        if isinstance(value, dict) and value:
            result.extend(paths_of(value, path + '.'))
        else:
            result.append(path)
    return result
