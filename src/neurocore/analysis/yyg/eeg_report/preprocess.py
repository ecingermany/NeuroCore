"""Signal preprocessing pipeline: detrend → 50Hz notch → 0.5-40Hz bandpass.

All filters use SOS + sosfiltfilt (zero-phase) for left-right alignment.
"""

import numpy as np
from scipy import signal as sig

from .config import (
    NOTCH_BAND, NOTCH_ORDER,
    BANDPASS_BAND, BANDPASS_ORDER,
    SAMPLE_RATE,
)


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """Remove linear trend from a 1D signal."""
    return sig.detrend(signal, type="linear")


def notch_filter(signal: np.ndarray, fs: int = SAMPLE_RATE,
                 band=None, order: int = NOTCH_ORDER) -> np.ndarray:
    """Apply 50Hz notch (bandstop) filter, zero-phase."""
    if band is None:
        band = NOTCH_BAND
    sos = sig.butter(order, band, btype="bandstop", fs=fs, output="sos")
    return sig.sosfiltfilt(sos, signal)


def bandpass_filter(signal: np.ndarray, fs: int = SAMPLE_RATE,
                     band=None, order: int = BANDPASS_ORDER) -> np.ndarray:
    """Apply bandpass filter, zero-phase."""
    if band is None:
        band = BANDPASS_BAND
    sos = sig.butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sig.sosfiltfilt(sos, signal)


def preprocess_signal(signal: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Full preprocessing chain: detrend → notch → bandpass.

    Input:  1D array [n_samples]
    Output: 1D array [n_samples] (same length, zero-phase)
    """
    signal = detrend_signal(signal)
    signal = notch_filter(signal, fs)
    signal = bandpass_filter(signal, fs)
    return signal


def preprocess_frames(frames: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Per-second independent preprocessing: detrend → notch → bandpass.

    Mirrors MATLAB's per-second filtfilt approach. Each row (256 samples = 1 second)
    is filtered independently, introducing edge effects at each second boundary.

    Input:  2D array [n_frames, frame_size]
    Output: 2D array [n_frames, frame_size] (same shape, zero-phase per row)
    """
    processed = np.empty_like(frames, dtype=np.float64)
    for i in range(frames.shape[0]):
        row = frames[i]
        row = detrend_signal(row)
        row = notch_filter(row, fs)
        row = bandpass_filter(row, fs)
        processed[i] = row
    return processed
