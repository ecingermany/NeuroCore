"""Spectral analysis: framing, windowing, FFT with energy compensation.

Mirrors the MATLAB apply_window_and_fft.m logic:
  frame → demean → window → FFT → normalize → energy compensate → DC clamp → AC double
"""

import numpy as np
from scipy import signal as sig

from .config import SAMPLE_RATE, FRAME_SIZE, TUKEY_ALPHA, MAX_DISPLAY_FREQ


def frame_signal(signal: np.ndarray, frame_size: int = FRAME_SIZE) -> np.ndarray:
    """Reshape 1D signal into [n_frames, frame_size] by integer seconds.

    Drops any trailing samples that don't fill a complete frame.
    """
    n_frames = len(signal) // frame_size
    return signal[:n_frames * frame_size].reshape(n_frames, frame_size)


def demean_frames(frames: np.ndarray) -> np.ndarray:
    """Subtract per-frame mean (removes per-second DC offset)."""
    return frames - frames.mean(axis=1, keepdims=True)


def get_window(window_type: str, n: int = FRAME_SIZE) -> np.ndarray:
    """Get window function by name.

    Supported: 'hanning', 'tukey' (alpha=0.1), 'hamming'.
    """
    if window_type == "hanning":
        return sig.windows.hann(n)
    elif window_type == "tukey":
        return sig.windows.tukey(n, TUKEY_ALPHA)
    elif window_type == "hamming":
        return sig.windows.hamming(n)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def compute_fft(frames: np.ndarray, window_type: str = "hanning",
                fs: int = SAMPLE_RATE) -> tuple:
    """Apply window + FFT + energy compensation to framed signal.

    Input:  frames [n_frames, frame_size]
    Output: (spectrum [n_frames, n_freqs], freq_axis [n_freqs])

    Processing chain (matches MATLAB apply_window_and_fft.m):
      1. Per-frame demean
      2. Apply window
      3. FFT → magnitude
      4. Base normalize: |X| / N
      5. Energy compensate: × 1/mean(window)
      6. DC clamp: max(bin0, 0.01)
      7. AC double: ×2 for single-sided spectrum (except DC)
    """
    frames = demean_frames(frames)
    window = get_window(window_type, frames.shape[1])
    windowed = frames * window[np.newaxis, :]

    N = frames.shape[1]
    raw_spectrum = np.abs(np.fft.rfft(windowed, axis=1))

    spectrum = raw_spectrum / N
    spectrum = spectrum * (1.0 / np.mean(window))
    spectrum[:, 0] = np.maximum(spectrum[:, 0], 0.01)
    spectrum[:, 1:] *= 2

    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)

    return spectrum, freq_axis


def compute_spectrogram(signal: np.ndarray, window_type: str = "hanning",
                        fs: int = SAMPLE_RATE,
                        max_freq: float = MAX_DISPLAY_FREQ) -> tuple:
    """Compute spectrogram from 1D signal.

    Returns: (spectrum [n_frames, n_freqs], freq_axis, time_axis [n_frames])
    """
    frames = frame_signal(signal, fs)
    spectrum, freq_axis = compute_fft(frames, window_type, fs)

    mask = freq_axis <= max_freq
    spectrum = spectrum[:, mask]
    freq_axis = freq_axis[mask]
    time_axis = np.arange(spectrum.shape[0])

    return spectrum, freq_axis, time_axis


def compute_band_power(spectrum: np.ndarray, freq_axis: np.ndarray,
                       freq_range: tuple) -> np.ndarray:
    """Compute average power (amplitude squared) in a frequency band per frame.

    Returns: [n_frames] array of mean power in the band.
    """
    mask = (freq_axis >= freq_range[0]) & (freq_axis <= freq_range[1])
    band_spectrum = spectrum[:, mask]
    return np.mean(band_spectrum ** 2, axis=1)
