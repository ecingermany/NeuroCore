"""MeasureR 580dee43 default N256 periodic-Hann replay profile."""
import numpy as np
from .._spectral import fft256
from ..data import SpectrumGrid, grid

PROFILE = 'measurer-580dee43-n256-hann-v1'
PANEL_BINS = {'Eye': (1, 2), 'Delta': (2, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
              'BetaL': (13, 19), 'BetaH': (19, 29), 'AC': (29, 57)}


def spectrum(raw):
    return fft256(raw, 'hann')


def band_rms(raw):
    p = spectrum(raw)
    return {name: np.sqrt(np.mean(p[..., lo:hi] ** 2, axis=-1)) for name, (lo, hi) in PANEL_BINS.items()}


def spectrum_grid(raw):
    a, f, t, profile, evidence = grid(raw, PROFILE, 'code and synthetic playback validation')
    return SpectrumGrid(spectrum(a), f, t, profile, evidence)


def surface_color_levels(current, previous):
    """Four-vertex means before palette integer conversion (rounding is unresolved)."""
    current, previous = np.asarray(current), np.asarray(previous)
    if current.ndim != 1 or current.shape != previous.shape or current.size < 2:
        raise ValueError('Expected two equal 1D spectra')
    return (current[1:] + current[:-1] + previous[1:] + previous[:-1]) / 4
