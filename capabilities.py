from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelCapabilities:
    tool_calling_supported: Optional[bool] = None
    multimodal_supported: Optional[bool] = None
    last_checked: Optional[float] = None


_LOCK = threading.Lock()
_CAPS = ModelCapabilities()


def set_capabilities(
    *,
    tool_calling_supported: Optional[bool] = None,
    multimodal_supported: Optional[bool] = None,
) -> ModelCapabilities:
    with _LOCK:
        if tool_calling_supported is not None:
            _CAPS.tool_calling_supported = tool_calling_supported
        if multimodal_supported is not None:
            _CAPS.multimodal_supported = multimodal_supported
        _CAPS.last_checked = time.time()
        return ModelCapabilities(
            tool_calling_supported=_CAPS.tool_calling_supported,
            multimodal_supported=_CAPS.multimodal_supported,
            last_checked=_CAPS.last_checked,
        )


def get_capabilities() -> ModelCapabilities:
    with _LOCK:
        return ModelCapabilities(
            tool_calling_supported=_CAPS.tool_calling_supported,
            multimodal_supported=_CAPS.multimodal_supported,
            last_checked=_CAPS.last_checked,
        )
