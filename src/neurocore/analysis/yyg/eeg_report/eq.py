"""EQ (Emotion Quotient) computation based on Alpha band asymmetry.

Algorithm (matches MATLAB EQ_main.m logic):
  1. Extract Alpha band (8-12Hz) average amplitude per frame for both channels
  2. Apply optional reduction factor to amplitudes
  3. Compute per-frame difference (A - B) over closed-eye window (frames 40-80)
  4. Sum differences: sum > 0 -> Depress (introvert/avoidance)
                      sum <= 0 -> Brightness (extrovert/approach)
"""

from dataclasses import dataclass
import numpy as np

from .config import ALPHA_BAND, CLOSED_EYE_START, CLOSED_EYE_END, EQ_REDUCTION_FACTOR


@dataclass
class EQResult:
    sum_diff: float               # Total sum of per-frame L-R differences
    direction: str                # "Depress" or "Brightness"
    per_frame_diff: np.ndarray    # [n_closed_eye_frames] L-R per frame
    A_alpha: np.ndarray            # Left brain alpha amplitude per frame (closed-eye)
    B_alpha: np.ndarray            # Right brain alpha amplitude per frame (closed-eye)
    positive_count: int            # Frames where A > B
    negative_count: int            # Frames where A < B
    reduction_factor: float        # Factor used


def compute_eq(
    A_spec: np.ndarray,
    B_spec: np.ndarray,
    freq_axis: np.ndarray,
    alpha_band: tuple = tuple(ALPHA_BAND),
    closed_eye_start: int = CLOSED_EYE_START,
    closed_eye_end: int = CLOSED_EYE_END,
    reduction_factor: float = EQ_REDUCTION_FACTOR,
) -> EQResult:
    """Compute EQ from two-channel spectra.

    Parameters
    ----------
    A_spec : [n_frames, n_freqs]   Left brain spectrum (amplitude)
    B_spec : [n_frames, n_freqs]   Right brain spectrum (amplitude)
    freq_axis : [n_freqs]          Frequency axis in Hz
    alpha_band : (low, high)       Alpha frequency band, default (8, 12)
    closed_eye_start : int         Closed-eye window start frame (0-indexed)
    closed_eye_end : int           Closed-eye window end frame (exclusive)
    reduction_factor : float       Amplitudes are divided by this before
                                   subtraction. Default 1.0 (no reduction).

    Returns
    -------
    EQResult
    """
    mask = (freq_axis >= alpha_band[0]) & (freq_axis <= alpha_band[1])

    A_alpha_full = np.mean(A_spec[:, mask], axis=1)
    B_alpha_full = np.mean(B_spec[:, mask], axis=1)

    A_alpha = A_alpha_full[closed_eye_start:closed_eye_end]
    B_alpha = B_alpha_full[closed_eye_start:closed_eye_end]

    if reduction_factor != 1.0:
        A_alpha = A_alpha / reduction_factor
        B_alpha = B_alpha / reduction_factor

    per_frame_diff = A_alpha - B_alpha
    sum_diff = float(np.sum(per_frame_diff))
    direction = "Depress" if sum_diff > 0 else "Brightness"
    positive_count = int(np.sum(per_frame_diff > 0))
    negative_count = int(np.sum(per_frame_diff < 0))

    return EQResult(
        sum_diff=sum_diff,
        direction=direction,
        per_frame_diff=per_frame_diff,
        A_alpha=A_alpha,
        B_alpha=B_alpha,
        positive_count=positive_count,
        negative_count=negative_count,
        reduction_factor=reduction_factor,
    )
