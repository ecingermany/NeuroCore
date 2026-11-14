"""Internal explicit FFT profiles. No YYG filters, band gains or interpolation."""
import numpy as np
from .data import blocks


def fft256(raw, window, pi=np.pi):
    a = blocks(raw, integer=True)
    n = np.arange(256)
    if window == 'hann':
        w = .5 - .5 * np.cos(2 * pi * n / 256)
    elif window == 'hamming':
        w = .54 - .46 * np.cos(2 * pi * n / 256)
    elif window == 'none':
        w = np.ones(256)
    else:
        raise ValueError('Unknown window')
    p = np.abs(np.fft.fft((a - 511) * (50 / 512) * w, axis=-1))[..., :128] / 16
    p[..., 0] = 0
    return p
