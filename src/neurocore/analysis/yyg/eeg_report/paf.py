"""PAF (Peak Alpha Frequency) computation.

Algorithm (matches MATLAB PAF_main.m + PAF_select.m):
  1. Uses Tukey window FFT (separate from Hanning used for EQ/3D viz)
  2. For closed-eye frames (40-80), detect peaks per frame in 0-29Hz
  3. Count peaks per frequency bin (no amplitude threshold since 2026-10-09:
     local maxima within 3-15Hz vote directly; legacy fixed threshold 2.0
     kept as fallback, see config.py PAF_VOTE_BAND for the validation stats)
  4. BRQ analysis: findpeaks on peak count histogram in 3-15Hz
  5. PAF_select: age-based selection from 3-14Hz candidates (tie -> higher freq)
  6. Also compute per-frame peak frequency for time series visualization
  7. Dominant-peak prominence gate on the mean spectrum (reliability judge)
"""

from dataclasses import dataclass
import numpy as np
from scipy import signal as sig

from .config import (
    PAF_FREQ_RANGE, PAF_BRQ_FREQ_RANGE, PAF_FULL_FREQ_MAX,
    PAF_PEAK_THRESHOLD, PAF_WINDOW_TYPE, CLOSED_EYE_START, CLOSED_EYE_END,
    PAF_VOTE_BAND,
    PAF_PROM_BAND, PAF_PROM_PEAK_RANGE,
)


@dataclass
class PAFResult:
    per_frame_paf_a: np.ndarray   # [n_ce_frames] peak freq per frame for A
    per_frame_paf_b: np.ndarray   # [n_ce_frames] peak freq per frame for B
    peak_counts_a: np.ndarray     # [30] accumulated peak counts (0-29Hz)
    peak_counts_b: np.ndarray     # [30]
    brq_peaks_a: np.ndarray       # BRQ peak frequencies (findpeaks on histogram)
    brq_peaks_b: np.ndarray
    paf_a: int                    # Final PAF for A (after age-based selection)
    paf_b: int                    # Final PAF for B
    age: int
    age_std: int                  # Age-standardized expected frequency
    # 峰突出度闸门（2026-10-09）：平均谱主峰频率与突出度（(峰高-鞍部)/鞍部×100%）
    dom_peak_a: int = None        # 左脑平均谱主峰频率 Hz，无峰为 None
    dom_ratio_a: float = None     # 左脑主峰突出度 %
    dom_peak_b: int = None        # 右脑平均谱主峰频率 Hz，无峰为 None
    dom_ratio_b: float = None     # 右脑主峰突出度 %


def age_to_std_freq(age: int) -> int:
    """Map age to expected peak alpha frequency (age_std).

    Updated from MATLAB with extended age ranges:
      5     -> 5
      6     -> 6
      7     -> 7
      8     -> 8
      9-10  -> 8
      11-14 -> 9
      15-39 -> 10
      40-50 -> 9
      51-60 -> 8
      61+   -> 7
    """
    if age < 5 or age > 100:
        raise ValueError(f"Age must be 5-100, got {age}")

    if 5 <= age <= 8:
        return age
    elif 9 <= age <= 10:
        return 8
    elif 11 <= age <= 14:
        return 9
    elif 15 <= age <= 39:
        return 10
    elif 40 <= age <= 50:
        return 9
    elif 51 <= age <= 60:
        return 8
    else:
        return 7


def _detect_peaks(data, threshold):
    """Find peaks above threshold using scipy.find_peaks (matches MATLAB findpeaks)."""
    locs, _ = sig.find_peaks(data)
    pks = data[locs]
    valid = pks > threshold
    return locs[valid]


def compute_peak_counts(spectrum, threshold=PAF_PEAK_THRESHOLD,
                        n_bins=PAF_FULL_FREQ_MAX + 1):
    """Accumulate peak counts per frequency bin across all frames.

    Equivalent to MATLAB loop:
      for sec = 1:numSeconds
          [pks, locs] = findpeaks(spectrum(sec, :));
          valid = pks > threshold;
          totalPeaksPerBin(locs(valid)) += 1;
      end

    Parameters
    ----------
    spectrum : [n_frames, n_bins]  Amplitude spectrum (0-29Hz)
    threshold : float             Noise floor
    n_bins : int                   Number of frequency bins (default 30)

    Returns
    -------
    np.ndarray [n_bins]  Peak count per frequency bin
    """
    counts = np.zeros(n_bins, dtype=int)
    for i in range(spectrum.shape[0]):
        locs = _detect_peaks(spectrum[i, :], threshold)
        for loc in locs:
            if 0 <= loc < n_bins:
                counts[loc] += 1
    return counts


def compute_peak_counts_band(spectrum, freq_axis, band=PAF_VOTE_BAND,
                             n_bins=PAF_FULL_FREQ_MAX + 1):
    """无阈值峰值计票（2026-10-09 下午定稿，PAF 默认计数路径）。

    规则：band（默认 3-15Hz）内的局部极大各计一票，不设任何幅度阈值。

    取代两条阈值路径（演进史与统计见 config.PAF_VOTE_BAND 注释）：
      - 固定绝对阈值 2.0：带内过滤率为 0（123-143 个局部极大近乎全过），
        历史作用只是挡 0-29Hz 全频计数的高频噪声，带内空转；
      - C2 动态阈值（同日上午方案）：幅度阈值对"平台型"alpha 有系统性盲区
        （gaoya 11-12Hz 抬升 21-48% 被中位数底线抹平）。

    真峰靠直方图时间维度累积浮现（40 帧反复出现的峰聚成 bin），噪声票随机
    散布相互抵消；结果可靠性由 compute_dominant_peak 突出度闸门独立评估。

    Parameters
    ----------
    spectrum : [n_frames, n_freqs]  幅度谱（闭眼期，任意频率轴）
    freq_axis : [n_freqs]           频率轴 Hz
    band : [low, high]              计票频段（默认 3-15，高频噪声主场 >15Hz 被排除）
    n_bins : int                    输出计数数组长度（默认 30）

    Returns
    -------
    np.ndarray [n_bins]  各频率 bin 峰值计数（band 之外恒为 0）
    """
    low, high = band
    counts = np.zeros(n_bins, dtype=int)
    mask = (freq_axis >= low) & (freq_axis <= high)
    band_spec = spectrum[:, mask]
    bin_idx = np.round(freq_axis[mask]).astype(int)
    valid_bins = (bin_idx >= 0) & (bin_idx < n_bins)

    for i in range(band_spec.shape[0]):
        locs = sig.find_peaks(band_spec[i, :])[0]
        for loc in locs:
            if valid_bins[loc]:
                counts[bin_idx[loc]] += 1
    return counts


def compute_dominant_peak(spectrum, freq_axis, band=PAF_PROM_BAND,
                          peak_range=PAF_PROM_PEAK_RANGE):
    """平均谱主峰突出度（PAF 可靠性闸门指标，2026-10-09；同日相邻谷鞍部修订）。

    判定逻辑（产品化为二元判定，突出度数值不在报告中显示）：
      1. peak_range（5-12Hz）内检出全部局部极大，取峰高最高者为候选主峰。
         峰数不设上限——多峰形态由突出度数学自然判定：峰越多越碎、峰间谷越浅，
         鞍部越高，最高峰突出度自动下降，无需按峰数写规则。
      2. 鞍部 = 相邻峰间谷（取代 scipy 延伸鞍部：后者水平线遇"比峰更高"的
         点才停，会穿过次峰延伸到远处深谷——双峰时次峰被吞掉、浅谷双尖被
         虚高成显著主峰，合成谱实验证实）：
           左谷 = 候选主峰与前一个 5-12Hz 峰之间的最小值（无前峰时延伸至
                  band 下沿，谷搜索可用 5-12Hz 以外的数据）
           右谷 = 同理向高频方向
           鞍部 = max(左谷, 右谷)
      3. 突出度 = (峰高 - 鞍部) / 鞍部 × 100%
      ≥30% 判"有明显稳定主峰"（定档锚点见 config 注释）。

    与 PAF 选择算法完全独立：不用阈值投票、不用直方图、不用年龄先验。
    平均谱本身是时间稳定性过滤器——只在少数帧出现的峰被平均稀释，
    反复出现的真峰才能幸存，因此单一数值同时编码"峰尖"与"时间稳定"。

    Parameters
    ----------
    spectrum : [n_frames, n_freqs]  幅度谱（闭眼期）
    freq_axis : [n_freqs]           频率轴 Hz
    band : [low, high]              谷搜索频段（默认 3-15，边缘谷可取到带边）
    peak_range : [low, high]        主峰搜索频段（默认 5-12，PAF 生理范围）

    Returns
    -------
    dict  {freq: 主峰 Hz, height: 峰高, base: 鞍部, ratio_pct: 突出度 %}
    None  5-12Hz 内未检出任何局部极大
    """
    low_b, high_b = band
    mask = (freq_axis >= low_b) & (freq_axis <= high_b)
    m = spectrum[:, mask].mean(axis=0)
    f = freq_axis[mask]

    all_locs = sig.find_peaks(m)[0]
    locs = [i for i in all_locs if peak_range[0] <= f[i] <= peak_range[1]]
    if not locs:
        return None

    p = max(locs, key=lambda i: m[i])
    left_peaks = [i for i in locs if f[i] < f[p]]
    right_peaks = [i for i in locs if f[i] > f[p]]
    lo_i = left_peaks[-1] if left_peaks else 0
    hi_i = right_peaks[0] if right_peaks else len(m) - 1
    left_valley = float(m[lo_i:p + 1].min())
    right_valley = float(m[p:hi_i + 1].min())
    base = max(left_valley, right_valley)
    if base <= 0:
        ratio = 999.0 if m[p] > 0 else 0.0
    else:
        ratio = (float(m[p]) - base) / base * 100.0
    return {
        "freq": int(round(f[p])),
        "height": float(m[p]),
        "base": base,
        "ratio_pct": ratio,
    }


def find_brq_peaks(peak_counts, freq_range=PAF_BRQ_FREQ_RANGE):
    """Find peaks in the peak count histogram (BRQ analysis).

    Equivalent to MATLAB:
      peaks_filtered = peak_counts(3:15+1);
      [pks, locs] = findpeaks(peaks_filtered);
      PAF_freq = locs + 2;  % convert to frequency

    Parameters
    ----------
    peak_counts : [30]   Full peak count array
    freq_range : [low, high]  BRQ analysis range (default 3-15)

    Returns
    -------
    np.ndarray  Peak frequencies (Hz)
    """
    low, high = freq_range
    freqs = np.arange(low, high + 1)
    data = peak_counts[low:high + 1]
    locs, _ = sig.find_peaks(data)
    return freqs[locs]


def _analyze_age_std(data, idx, frequencies, age_std):
    """Recursive PAF selection (analyze_age_std_new.m equivalent).

    1. Candidate offsets: [-2, -1, 0, +1] from age_std index
    2. Find candidate with highest peak count
    3. Tie -> select higher frequency
    4. If selected freq > 4, return it
    5. If <= 4, remove from candidates and recurse
    6. If no candidates remain, return age_std
    """
    offsets = [-2, -1, 0, 1]
    candidate_indices = []
    for offset in offsets:
        ci = idx + offset
        if 0 <= ci < len(frequencies):
            candidate_indices.append(ci)

    if len(candidate_indices) == 0:
        return age_std

    candidate_indices = np.array(candidate_indices)
    candidate_x = frequencies[candidate_indices]
    candidate_y = data[candidate_indices]

    max_y = np.max(candidate_y)
    max_mask = candidate_y == max_y

    if np.sum(max_mask) > 1:
        max_x_vals = candidate_x[max_mask]
        selected_local = np.argmax(max_x_vals)
        selected_idx = np.where(max_mask)[0][selected_local]
    else:
        selected_idx = np.where(max_mask)[0][0]

    selected_x = candidate_x[selected_idx]

    if selected_x > 4:
        return selected_x

    remaining = np.delete(np.arange(len(candidate_x)), selected_idx)
    if len(remaining) == 0:
        return age_std

    remaining_data = data[candidate_indices[remaining]]
    remaining_freqs = frequencies[candidate_indices[remaining]]

    age_std_match = np.where(remaining_freqs == age_std)[0]
    age_std_idx = age_std_match[0] if len(age_std_match) > 0 else 0

    return _analyze_age_std(remaining_data, age_std_idx, remaining_freqs, age_std)


def select_paf(peak_counts, age, freq_range=PAF_FREQ_RANGE):
    """Age-based PAF selection (PAF_select.m equivalent).

    Parameters
    ----------
    peak_counts : [30]  Full peak count array (0-29Hz)
    age : int           Subject age (5-100)
    freq_range : [low, high]  Selection range (default 3-14)

    Returns
    -------
    int  Selected PAF frequency in Hz
    """
    low, high = freq_range
    frequencies = np.arange(low, high + 1)
    data = peak_counts[low:high + 1]

    age_std = age_to_std_freq(age)

    idx_match = np.where(frequencies == age_std)[0]
    idx = idx_match[0] if len(idx_match) > 0 else \
        np.argmin(np.abs(frequencies - age_std))

    result = _analyze_age_std(data, idx, frequencies, age_std)
    return int(result)


def compute_per_frame_paf(spectrum, freq_axis, threshold=PAF_PEAK_THRESHOLD,
                           freq_range=PAF_FREQ_RANGE):
    """Per-frame peak frequency for time series visualization.

    For each frame, finds peaks above threshold in the 3-14Hz range.
    Returns the frequency of the highest peak. If no peak above threshold,
    falls back to argmax (highest amplitude in range).

    Parameters
    ----------
    spectrum : [n_frames, n_freqs]  Full spectrum
    freq_axis : [n_freqs]          Frequency axis in Hz
    threshold : float               Noise floor threshold
    freq_range : [low, high]        Search range

    Returns
    -------
    np.ndarray [n_frames]  Peak frequency per frame (Hz)
    """
    low, high = freq_range
    mask = (freq_axis >= low) & (freq_axis <= high)
    band_spec = spectrum[:, mask]
    band_freqs = freq_axis[mask]

    paf_per_frame = np.zeros(spectrum.shape[0])
    for i in range(spectrum.shape[0]):
        data = band_spec[i, :]
        locs = _detect_peaks(data, threshold)
        if len(locs) > 0:
            paf_per_frame[i] = band_freqs[locs[np.argmax(data[locs])]]
        else:
            paf_per_frame[i] = band_freqs[np.argmax(data)]

    return paf_per_frame


def compute_paf(
    A_spec, B_spec, freq_axis, age,
    closed_eye_start=CLOSED_EYE_START,
    closed_eye_end=CLOSED_EYE_END,
    peak_threshold=PAF_PEAK_THRESHOLD,
    paf_freq_range=PAF_FREQ_RANGE,
    brq_freq_range=PAF_BRQ_FREQ_RANGE,
    full_freq_max=PAF_FULL_FREQ_MAX,
    use_legacy_threshold=False,
):
    """Compute PAF from two-channel spectra (Tukey FFT).

    Parameters
    ----------
    A_spec : [n_frames, n_freqs]   Left brain spectrum (Tukey FFT)
    B_spec : [n_frames, n_freqs]   Right brain spectrum (Tukey FFT)
    freq_axis : [n_freqs]          Frequency axis in Hz
    age : int                      Subject age (5-100)
    closed_eye_start : int         Closed-eye window start frame (0-indexed)
    closed_eye_end : int           Closed-eye window end frame (exclusive)
    peak_threshold : float         旧版固定绝对阈值（仅 use_legacy_threshold=True 时使用）
    paf_freq_range : [low, high]   PAF selection range
    brq_freq_range : [low, high]   BRQ analysis range
    full_freq_max : int             Max frequency for peak counting
    use_legacy_threshold : bool    False（默认）= 无阈值计票（3-15Hz 带内局部极大，
                                   2026-10-09 定稿）；True = 回退旧版固定阈值 2.0

    Returns
    -------
    PAFResult
    """
    freq_mask = freq_axis <= full_freq_max
    fax_ce = freq_axis[freq_mask]

    A_ce = A_spec[closed_eye_start:closed_eye_end, :][:, freq_mask]
    B_ce = B_spec[closed_eye_start:closed_eye_end, :][:, freq_mask]

    if use_legacy_threshold:
        # 旧版固定绝对阈值路径（MATLAB 遗产，保留对照/回退）
        peak_counts_a = compute_peak_counts(A_ce, peak_threshold)
        peak_counts_b = compute_peak_counts(B_ce, peak_threshold)
    else:
        # 无阈值计票（2026-10-09 定稿）：3-15Hz 带内局部极大直接计票
        peak_counts_a = compute_peak_counts_band(A_ce, fax_ce)
        peak_counts_b = compute_peak_counts_band(B_ce, fax_ce)

    brq_peaks_a = find_brq_peaks(peak_counts_a, brq_freq_range)
    brq_peaks_b = find_brq_peaks(peak_counts_b, brq_freq_range)

    paf_a = select_paf(peak_counts_a, age, paf_freq_range)
    paf_b = select_paf(peak_counts_b, age, paf_freq_range)

    per_frame_a = compute_per_frame_paf(
        A_spec[closed_eye_start:closed_eye_end], freq_axis,
        peak_threshold, paf_freq_range,
    )
    per_frame_b = compute_per_frame_paf(
        B_spec[closed_eye_start:closed_eye_end], freq_axis,
        peak_threshold, paf_freq_range,
    )

    age_std = age_to_std_freq(age)

    # 峰突出度闸门：平均谱主峰形态，独立评估 PAF 结果可靠性
    dom_a = compute_dominant_peak(A_ce, fax_ce)
    dom_b = compute_dominant_peak(B_ce, fax_ce)

    return PAFResult(
        per_frame_paf_a=per_frame_a,
        per_frame_paf_b=per_frame_b,
        peak_counts_a=peak_counts_a,
        peak_counts_b=peak_counts_b,
        brq_peaks_a=brq_peaks_a,
        brq_peaks_b=brq_peaks_b,
        paf_a=paf_a,
        paf_b=paf_b,
        age=age,
        age_std=age_std,
        dom_peak_a=dom_a["freq"] if dom_a else None,
        dom_ratio_a=dom_a["ratio_pct"] if dom_a else None,
        dom_peak_b=dom_b["freq"] if dom_b else None,
        dom_ratio_b=dom_b["ratio_pct"] if dom_b else None,
    )
