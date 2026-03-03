from __future__ import annotations


_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def is_env_flag_enabled(name: str, default: str = "false") -> bool:
    value = default
    from os import getenv

    env_value = getenv(name)
    if env_value is not None:
        value = env_value
    return value.strip().lower() in _TRUTHY_VALUES
