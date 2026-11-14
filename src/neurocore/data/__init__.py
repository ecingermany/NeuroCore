"""Public data contracts and independent, strict sample validation."""
from dataclasses import dataclass
import numpy as np


def blocks(values, size=256, integer=False, maximum=1023):
    """Return copied float64 blocks with the final dimension equal to size.

    This package rejects invalid inputs explicitly; this is a package policy,
    not an assertion about how original software handles malformed files.
    """
    a = np.asarray(values)
    if a.ndim == 0 or a.shape[-1] != size or a.size == 0:
        raise ValueError('Expected nonempty samples with final dimension %d' % size)
    if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a) or not np.isfinite(a).all():
        raise ValueError('Samples must be finite numbers')
    if integer and ((a != np.trunc(a)).any() or (a < 0).any() or (a > maximum).any()):
        raise ValueError('Expected integer ADC codes in 0..%d' % maximum)
    return a.astype(np.float64, copy=True)


@dataclass
class SpectrumGrid:
    """Single-channel amplitude grid. Units are uncalibrated internal units."""
    amplitude: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    profile: str
    evidence: str
    units: str = 'internal amplitude (uncalibrated)'


def grid(values, profile, evidence, sample_rate=256):
    a = blocks(values, integer=True)
    if a.ndim != 2:
        raise ValueError('Expected [seconds, 256] for a surface')
    return a, np.arange(128) * sample_rate / 256, np.arange(1, len(a) + 1), profile, evidence
