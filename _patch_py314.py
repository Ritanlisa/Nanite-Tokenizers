"""
Monkey-patch pydantic v1 for Python 3.14+ compatibility.

Python 3.14 implements PEP 649 (Deferred Evaluation of Annotations),
which defers annotation evaluation and stores them as __annotate_func__
in the class namespace during class creation, rather than populating
__annotations__ directly. pydantic v1's ModelMetaclass relies on
__annotations__ being present in the namespace dict, causing a
ConfigError when it cannot infer field types.

This patch detects the missing __annotations__ and uses type.__new__
to create a temporary class that properly resolves annotations via
PEP 649, then injects them back into the namespace before pydantic
processes the class.

Safe to import on any Python version (no-op on < 3.14).
"""

import sys

if sys.version_info >= (3, 14):
    import inspect

    from pydantic.v1 import main as _pydantic_main

    _original_new = _pydantic_main.ModelMetaclass.__new__

    def _patched_new(mcs, name, bases, namespace, **kwargs):
        if '__annotate_func__' in namespace and '__annotations__' not in namespace:
            try:
                ns_copy = dict(namespace)
                temp_cls = type.__new__(mcs, name, bases, ns_copy)
                resolved = inspect.get_annotations(temp_cls)
                if resolved:
                    namespace['__annotations__'] = resolved
            except Exception:
                pass
        return _original_new(mcs, name, bases, namespace, **kwargs)

    _pydantic_main.ModelMetaclass.__new__ = staticmethod(_patched_new)
