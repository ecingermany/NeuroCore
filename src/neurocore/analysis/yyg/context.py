"""Host-provided business copy, isolated by execution context (no cache/network I/O)."""
from contextlib import contextmanager
from contextvars import ContextVar

_library = ContextVar('neurocore_yyg_advice_library', default=None)


def get_advice_library():
    return _library.get()


@contextmanager
def advice_library(library):
    """Supply the existing host AdviceLibrary when legacy interpretations are wanted."""
    token = _library.set(library)
    try:
        yield
    finally:
        _library.reset(token)
