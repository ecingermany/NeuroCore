"""BrainSee default no-window display profile: strong static evidence, not pixel parity."""
from .._spectral import fft256
from ..data import SpectrumGrid, grid

PROFILE = 'brainsee-n256-rect-static-v1'


def spectrum(raw):
    return fft256(raw, 'none')


def spectrum_grid(raw):
    a, f, t, profile, evidence = grid(raw, PROFILE, 'static inference; no pixel-level validation')
    return SpectrumGrid(spectrum(a), f, t, profile, evidence)
