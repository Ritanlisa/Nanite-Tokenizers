import os
import sys
import tempfile
from typing import Callable, Optional

profiler = None
_profile_path: Optional[str] = None


def start_profiler(path: Optional[str] = None) -> Optional[str]:
    """Start a LineProfiler if available. Returns path where stats will be saved."""
    global profiler, _profile_path
    try:
        from line_profiler import LineProfiler  # type: ignore

        profiler = LineProfiler()
        _profile_path = path or os.path.join(tempfile.gettempdir(), "rag_line_profile.lprof")
        return _profile_path
    except Exception:
        profiler = None
        _profile_path = None
        return None


def stop_profiler() -> Optional[str]:
    """Dump profiler stats to disk (if started) and print brief summary. Returns path or None."""
    global profiler, _profile_path
    if profiler is None:
        return None
    try:
        # dump raw stats for later visualization
        if _profile_path:
            try:
                profiler.dump_stats(_profile_path)
            except Exception:
                pass
        # print human readable summary to stdout
        try:
            profiler.print_stats(stream=sys.stdout)
        except Exception:
            pass
    finally:
        profiler = None
    return _profile_path


def profile_if_enabled(func: Callable) -> Callable:
    """Decorator: if a profiler is started, run the function under it; otherwise call normally."""

    def wrapper(*args, **kwargs):
        global profiler
        if profiler is None:
            return func(*args, **kwargs)
        # use runcall so line_profiler collects line-level stats
        try:
            return profiler.runcall(func, *args, **kwargs)
        except Exception:
            # ensure original exception propagation
            raise

    try:
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
    except Exception:
        pass
    return wrapper
