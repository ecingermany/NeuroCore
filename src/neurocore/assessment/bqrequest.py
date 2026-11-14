"""BQRequest f87df6b1 report mathematics from decoded O1/C/O2 sample blocks.

Uses the exact historical reference PI=3.141592654. This module never loads a
vendor DLL, decrypts a BFN, contacts a server, or reads personal files on import.
"""
import numpy as np
from .._spectral import fft256
from .._feedback import compute_srq
from ..data import blocks

PROFILE = 'bqrequest-f87df6b1-n256-hamming-v1'
BOUNDS = ((1, 4), (4, 8), (8, 13), (12, 16), (16, 21), (21, 31))


def spectrum(raw):
    return fft256(raw, 'hamming', pi=3.141592654)


def age_parameters(age):
    if isinstance(age, bool) or not isinstance(age, (int, np.integer)) or not 4 <= age <= 99:
        raise ValueError('Supported age is an integer in 4..99')
    table = ((5, 5, 6), (6, 6, 5), (7, 7, 4), (9, 8, 4), (14, 9, 3),
             (45, 10, 2), (59, 9, 3), (69, 8, 4), (79, 7, 5), (99, 6, 6))
    return next((f, c) for limit, f, c in table if age <= limit)


def _transform(z):
    return 100 * (1 + np.log10(z) * np.where(z <= 1, 1, .1))


def compute_report(static_stages, age, percentages, weights):
    """Three [30,2,256] O1/C/O2 stages and saved/reconstructed training P/W.

    No silent fill, clamping of internal scores, or inference of missing stages.
    Nonpositive ratio/log inputs are rejected as package policy.
    """
    ref, c = age_parameters(age)
    raw = blocks(static_stages, integer=True)
    if raw.shape != (3, 30, 2, 256):
        raise ValueError('Expected [3 stages, 30 seconds, 2 channels, 256 samples]')
    srq = compute_srq(percentages, weights)
    spectra = spectrum(raw)
    bands = np.stack([spectra[..., lo:hi].mean(-1) for lo, hi in BOUNDS], -1)
    means = np.stack([spectra[..., lo+1:hi+1].mean(axis=(1, 3)) for lo, hi in BOUNDS], -1)
    if (bands[..., 3] <= 0).any() or (means <= 0).any():
        raise ValueError('Undefined report: zero band denominator or nonpositive log input')
    rate = np.mean(bands[..., 1] / bands[..., 3], axis=(0, 1))
    denominator = bands[..., 3] + bands[..., 4] - spectra[..., 12] / 4
    if (denominator <= 0).any():
        raise ValueError('Undefined alpha/low-beta ratio')
    q_activity = np.mean(bands[..., 2] / denominator, axis=(0, 1))
    if (rate <= 0).any() or (q_activity <= 0).any():
        raise ValueError('Undefined logarithmic report')
    atq = _transform(c / rate) - 10 * np.log(np.maximum(1, means[2, :, 0] + means[2, :, 5]))
    acq = _transform(.3 * c / q_activity) - 10 * np.log(rate)
    counts = np.zeros((2, 31), dtype=int)
    for frame in spectra[1]:
        for channel in range(2):
            for k in range(1, 30):
                if frame[channel, k] > frame[channel, k-1] and frame[channel, k+1] < frame[channel, k]:
                    counts[channel, k] += 1
    peaks = []
    for row in counts:
        lo, hi = max(ref-2, 5), min(ref+2, 12)
        best, value, decline = lo, row[lo], 0
        for k in range(lo+1, hi):
            if row[k] < value:
                decline += 1
                best = k - decline
            else:
                decline, best, value = 0, k, row[k]
        peaks.append(best)
    peak_amplitudes = spectra[1].mean(0)[np.arange(2), peaks]
    if (peak_amplitudes <= 0).any():
        raise ValueError('Undefined BRQ peak amplitude')
    amplitude_ratio = 100 * (peak_amplitudes - means[1].mean(-1)) / peak_amplitudes
    attenuation = 100 * (means[1, :, 2] - means[2, :, 2]) / means[1, :, 2]
    frequency_score = np.minimum(100, (np.asarray(peaks) * 100 / ref).astype(int))
    brq = .7 * frequency_score + .15 * amplitude_ratio + .15 * attenuation
    eq_curve = bands[1, :, 0, 2] - bands[1, :, 1, 2]
    eq = 50 - abs(np.mean(eq_curve)) + (sum(brq) + sum(atq)) / 8
    left = raw[:, :, 0] - raw[:, :, 0].mean(-1)[..., None]
    right = raw[:, :, 1] - raw[:, :, 1].mean(-1)[..., None]
    correlation = np.sum(left * right, -1) / (np.sqrt(np.sum(left*left, -1) * np.sum(right*right, -1)) + 1e-20)
    synchrony = float(correlation.mean())
    ratio = float(np.mean(means[[2, 1], 0] / means[[2, 1], 1]))
    symmetry = 50 * (ratio if ratio <= 1 else 1 / ratio)
    cq = symmetry + 25 * (synchrony + 1)
    anti_stress = 100 - means[2, :, 0] - means[2, :, 5] - .2 * (100 - atq)
    bq = (sum(brq) + srq + sum(atq) + sum(acq) + eq + sum(anti_stress) + cq) / 11
    if any((spectra[..., lo+1:hi+1] <= 0).any() for lo, hi in BOUNDS):
        raise ValueError('Undefined log spectrum')
    logs = np.stack([np.log(spectra[..., lo+1:hi+1]).mean(axis=(1, 3)) for lo, hi in BOUNDS], -1)
    slopes = np.array([(logs[..., 0]-logs[..., 5])/5,
                       (logs[..., 0]-logs[..., 2])/2, (logs[..., 2]-logs[..., 5])/3])
    mean_o2 = spectra[2].mean(0)
    absolute = mean_o2[:, 13:17].mean(-1) + mean_o2[:, 17:22].mean(-1) - mean_o2[:, 12]/4
    relative = spectra[1].mean(0)[:, 9:14].mean(-1)
    return dict(profile=PROFILE, age=int(age), BRQ=brq, SRQ=srq, Rate=rate, AoverBl=q_activity,
                ATQ=atq, ACQ=acq, EQ=float(eq), AntiStress=anti_stress,
                Synchrony=synchrony, Symmetry=symmetry, CQ=cq, BQ=float(bq),
                Tension=means[2, :, 0], Distraction=means[2, :, 5], peak_frequencies=np.array(peaks),
                EQ_curve=eq_curve, bands=bands, mean_bands=means, spectra=spectra,
                AbsoluteStrength=absolute, RelativeStrength=relative,
                LogDifference=float(np.log(relative[1])-np.log(relative[0])),
                ArithmeticDifference=float((relative[1]-relative[0])/sum(relative)),
                activity_label='Negative' if q_activity[1] < q_activity[0] else 'Positive',
                emotion_label='우울' if eq_curve.mean() >= 0 else '명랑',
                LogBand=logs, Slope=slopes,
                BrainCondition=float(np.sum(percentages)), MeanValue=float(np.mean(percentages)),
                MaximumDeviation=float(np.ptp(percentages)), StandardDeviation=float(np.std(percentages)))
