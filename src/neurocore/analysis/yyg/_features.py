"""Extracted YYG algorithms; numeric behavior preserved. See provenance.json.
Historical units/interpretations in source are not physical calibration.
"""


from .context import get_advice_library as _get_advice_library

import sys

import numpy as np

from dataclasses import dataclass, field

from pathlib import Path

from typing import Optional

from .eeg_report.loader import load_eeg_txt

from .eeg_report.preprocess import preprocess_signal, preprocess_frames

from .eeg_report.spectral import frame_signal, compute_fft

from .eeg_report.config import (
    SAMPLE_RATE, MAX_DISPLAY_FREQ,
    CLOSED_EYE_START, CLOSED_EYE_END,
    ALPHA_BAND, EQ_REDUCTION_FACTOR,
    PAF_FREQ_RANGE, PAF_FULL_FREQ_MAX,
    PAF_PEAK_THRESHOLD, PAF_WINDOW_TYPE,
    PAF_PROM_SIGNIFICANT,
)

from .eeg_report.eq import compute_eq

from .eeg_report.paf import compute_paf

TENSION_BAND = (0, 3)

SCALING_FACTORS = {
    (0, 2): 0.40,    # Eye
    (2, 4): 0.31,    # Delta
    (4, 8): 0.24,    # Theta
    (8, 12): 0.21,   # Alpha
    (12, 20): 0.20,  # LowBeta
    (20, 30): 0.15,  # HighBeta
}

TIME_SEGMENTS = [
    ("full",       0,   None),         # 全程
    ("eyes_open_1", 0,   CLOSED_EYE_START),   # 睁眼1: 0-40s
    ("eyes_closed", CLOSED_EYE_START, CLOSED_EYE_END),  # 闭眼: 40-80s
    ("eyes_open_2", CLOSED_EYE_END,   None),         # 睁眼2: 80-120s
]

TENSION_SCALING_AVG = 0.355

TENSION_ABS_THRESHOLD_RAW = 56.3

TENSION_ABS_THRESHOLD_SCALED = 20.0

THETA_EMOTION_BAND = (4, 7)

THETA_EMOTION_SCALING = 0.24

THETA_EMOTION_THRESHOLD_RAW = 54.2

THETA_EMOTION_THRESHOLD_SCALED = 13.0

THETA_DROP_RATIO = 0.4

THETA_DROP_MAX_DURATION = 4

THETA_DROP_ABS_THRESHOLD = 15.0

THETA_DROP_BASELINE_TOLERANCE = 1.1

def get_scaling_array(freq_axis: np.ndarray) -> np.ndarray:
    """为每个频率 bin 生成对应的缩放因子（原始版，用于算法阈值）。"""
    scale = np.ones_like(freq_axis, dtype=float)
    for (fmin, fmax), factor in SCALING_FACTORS.items():
        mask = (freq_axis >= fmin) & (freq_axis < fmax)
        scale[mask] = factor
    scale[freq_axis >= MAX_DISPLAY_FREQ] = 0.0
    return scale

@dataclass
class ChannelSpectrum:
    """单通道的频谱数据。"""
    spectrum: np.ndarray       # [n_frames, n_freqs] amplitude
    freq_axis: np.ndarray     # [n_freqs] Hz
    time_axis: np.ndarray     # [n_frames] seconds
    spectrum_scaled: np.ndarray

@dataclass
class FeatureResult:
    """单个特征的计算结果。"""
    name: str                       # 特征名称
    description: str                # 简短描述
    channel_a: dict                 # Channel A 结果
    channel_b: dict                 # Channel B 结果
    interpretation: str = ""        # 综合解读
    references: list = field(default_factory=list)  # 相关文献
    paf_result: object = None

def _tension_closed_rise_detail(ch_a: dict, ch_b: dict) -> tuple:
    """闭眼段相对睁眼段的紧张上升程度：2=明显升高 / 1=略升 / 0=持平或下降。

    睁眼基线取两段睁眼（睁眼1 / 睁眼2）中较高者。
    明显升高（任一侧脑满足其一）：
    - 闭眼超标帧数比睁眼基线多 ≥5 帧；
    - 闭眼平均波幅比睁眼基线高 ≥10%。

    返回 (level, evidence)，evidence 为 level=2 时的量化依据文本。
    """
    level = 0
    for ch in (ch_a, ch_b):
        segs = ch.get("segments", {})
        o1 = segs.get("eyes_open_1", {})
        o2 = segs.get("eyes_open_2", {})
        ce = segs.get("eyes_closed", {})
        o_exceed = max(o1.get("method_a_exceed_count", 0),
                       o2.get("method_a_exceed_count", 0))
        o_mean = max(o1.get("method_a_mean_amplitude_raw") or 0,
                     o2.get("method_a_mean_amplitude_raw") or 0)
        frame_diff = ce.get("method_a_exceed_count", 0) - o_exceed
        c_mean = ce.get("method_a_mean_amplitude_raw") or 0
        if frame_diff >= 5:
            return 2, f"闭眼超标帧数较睁眼多 {frame_diff} 帧"
        if o_mean > 0 and c_mean >= o_mean * 1.10 - 1e-9:
            pct = round((c_mean - o_mean) / o_mean * 100)
            return 2, f"闭眼平均波幅较睁眼高约 {pct}%"
        if frame_diff > 0 or (o_mean > 0 and c_mean > o_mean):
            level = max(level, 1)
    return level, ""

def _tension_closed_rise_level(ch_a: dict, ch_b: dict) -> int:
    return _tension_closed_rise_detail(ch_a, ch_b)[0]

def compute_physiological_tension(ch_a: ChannelSpectrum,
                                  ch_b: ChannelSpectrum,
                                  age: int = 30) -> FeatureResult:
    """特征 #1: 0-3 Hz 生理紧张（Physiological Tension）

    方法 A — 绝对幅值法：
        计算 0-3 Hz 的缩放后 amplitude，统计超过 20 µV 的帧数占比

    方法 B — 相对功率法：
        计算 0-3 Hz amplitude / 0-30 Hz 总 amplitude 的比值

    生理基础：
        焦虑/紧张 → 眨眼力度增大、频率升高 → FP1/FP2 的 0-3 Hz Delta 升高
        眼眨功率谱峰值在 0.5-3 Hz，被前额电极捕获
        在数据质量控制前提下，此信号即为生理紧张的指标

    文献见: feature_01_physiological_tension.md
    """
    threshold_raw = get_age_adjusted_tension_threshold(age)

    result = FeatureResult(
        name="Physical Stress（生理紧张）",
        description="0-3 Hz Delta 频段分析",
        channel_a={},
        channel_b={},
        references=["feature_01_physiological_tension.md"],
    )

    for label, ch in [("A", ch_a), ("B", ch_b)]:
        freq_mask = (ch.freq_axis >= TENSION_BAND[0]) & (ch.freq_axis <= TENSION_BAND[1])
        total_mask = ch.freq_axis <= MAX_DISPLAY_FREQ

        # 预计算逐帧值（全程）
        band_amplitude_raw = np.mean(ch.spectrum[:, freq_mask], axis=1)

        # 按时间段分别计算
        segments = {}
        for seg_name, start, end in TIME_SEGMENTS:
            s = start
            e = end if end is not None else len(band_amplitude_raw)
            seg_amp = band_amplitude_raw[s:e]

            if len(seg_amp) == 0:
                continue

            seg_mean_raw = float(np.mean(seg_amp))
            exceed_count = int(np.sum(seg_amp > threshold_raw))
            total = len(seg_amp)

            segments[seg_name] = {
                "method_a_threshold_raw": threshold_raw,
                "method_a_threshold_scaled": TENSION_ABS_THRESHOLD_SCALED,
                "method_a_mean_amplitude_raw": round(seg_mean_raw, 3),
                "method_a_mean_amplitude_scaled": round(seg_mean_raw * TENSION_SCALING_AVG, 3),
                "method_a_mean_to_threshold_ratio": round(seg_mean_raw / threshold_raw, 3),
                "method_a_exceed_count": exceed_count,
                "method_a_total_frames": total,
                "method_a_exceed_ratio": round(exceed_count / total, 4),
            }

        ch_result = {
            "threshold_raw": threshold_raw,
            "threshold_scaled": TENSION_ABS_THRESHOLD_SCALED,
            "segments": segments,
            "delta_open1_exceed": segments.get("eyes_open_1", {}).get("method_a_exceed_count", 0),
            "delta_closed_exceed": segments.get("eyes_closed", {}).get("method_a_exceed_count", 0),
            "delta_open2_exceed": segments.get("eyes_open_2", {}).get("method_a_exceed_count", 0),
            "delta_change_1": segments.get("eyes_closed", {}).get("method_a_exceed_count", 0) - segments.get("eyes_open_1", {}).get("method_a_exceed_count", 0),
            "delta_change_2": segments.get("eyes_closed", {}).get("method_a_exceed_count", 0) - segments.get("eyes_open_2", {}).get("method_a_exceed_count", 0),
        }

        if label == "A":
            result.channel_a = ch_result
        else:
            result.channel_b = ch_result

    # 综合解读（基于全程数据）
    a_ratio = result.channel_a["segments"]["full"]["method_a_exceed_ratio"]
    b_ratio = result.channel_b["segments"]["full"]["method_a_exceed_ratio"]
    a_mean_ratio = result.channel_a["segments"]["full"]["method_a_mean_to_threshold_ratio"]
    b_mean_ratio = result.channel_b["segments"]["full"]["method_a_mean_to_threshold_ratio"]

    parts = []
    if max(a_ratio, b_ratio) > 0.6:
        parts.append("紧张信号严重（超过60%的时间超过阈值）")
    elif max(a_ratio, b_ratio) > 0.3:
        parts.append("紧张信号明显（超过30%的时间超过阈值）")
    elif max(a_ratio, b_ratio) > 0.1:
        parts.append("存在轻度紧张信号（10-30%的时间超过阈值）")
    else:
        parts.append("紧张信号不明显（少于10%的时间超过阈值）")

    if max(a_mean_ratio, b_mean_ratio) > 2.0:
        parts.append("整体平均波幅达到阈值的两倍，持续高压状态")
    elif max(a_mean_ratio, b_mean_ratio) > 1.5:
        parts.append("整体平均波幅为阈值的1.5倍，紧张信号明显")
    elif max(a_mean_ratio, b_mean_ratio) > 1.0:
        parts.append("整体平均波幅超过阈值")

    # 睁闭眼对比解读（明显升高的判定与建议触发一致：超标帧多≥5 或平均波幅高≥10%）
    # 明显升高时放最前并附量化依据——它是触发建议的原因，避免与后面的正常项矛盾
    rise, rise_evidence = _tension_closed_rise_detail(result.channel_a, result.channel_b)
    if rise == 2:
        evidence = f"（{rise_evidence}）" if rise_evidence else ""
        parts.insert(0, f"闭眼时紧张信号较睁眼时明显增加{evidence}，放松效果不理想")
    elif rise == 1:
        parts.append("闭眼时紧张信号较睁眼时略有增加，未达到明显程度")
    else:
        parts.append("闭眼时紧张信号较睁眼时减少或持平，放松状态正常")

    result.interpretation = "；".join(parts)

    return result

def detect_theta_drop_events(amps: np.ndarray,
                             baseline_limit: float = None,
                             drop_ratio: float = THETA_DROP_RATIO,
                             max_duration: int = THETA_DROP_MAX_DURATION,
                             abs_threshold: float = THETA_DROP_ABS_THRESHOLD) -> list:
    """检测 Theta 波幅骤降事件（反映压力激素对短期记忆的瞬时抑制）。

    判定条件（全部满足才记为一次骤降事件）：
      1. 基线 = 前两秒最大值（覆盖单秒骤降与缓坡式两秒下降）
         且基线 < baseline_limit（年龄阈值×容差）——基线超标的秒数
         已由主分析的超标占比报告，骤降分析只关注健康波形的突降
      2. 当前秒 < 基线 × drop_ratio（降幅超过 60%）
      3. 当前秒绝对值 ≤ abs_threshold（必须降到很低才算，单纯波动不算）
      4. 持续 ≤ max_duration 秒（更长的下降属于状态变化而非瞬时抑制）

    骤降结束后，从恢复点用新的两秒窗口作为基线继续检测；
    达到 max_duration 仍未恢复的，从 start+max_duration 处重新起算，
    避免长时间低幅值被误判为连续多次事件。
    """
    events = []
    i = 2
    while i < len(amps):
        baseline = float(np.max(amps[i - 2:i]))
        curr = amps[i]
        in_range = baseline_limit is None or baseline < baseline_limit
        if baseline > 0 and in_range and curr < baseline * drop_ratio and curr <= abs_threshold:
            start = i
            while (i < len(amps) and amps[i] < baseline * drop_ratio
                   and amps[i] <= abs_threshold and (i - start) < max_duration):
                i += 1
            duration = i - start
            min_amp = float(np.min(amps[start:i]))
            events.append({
                "start_sec": int(start),
                "end_sec": int(i),
                "duration": int(duration),
                "baseline_mV": round(baseline, 3),
                "min_amp_mV": round(min_amp, 3),
                "drop_pct": round((1 - min_amp / baseline), 3) if baseline > 0 else 0.0,
            })
        else:
            i += 1
    return events

def compute_theta_emotion(ch_a: ChannelSpectrum,
                           ch_b: ChannelSpectrum,
                           age: int = 30) -> FeatureResult:
    """特征 #6: Theta 压抑情绪（Theta Emotional Suppression）

    方法 A — 绝对幅值法：
        计算 4-7 Hz 的原始 amplitude，统计超过 54.2 µV（缩放后 13 µV）的帧数占比

    生理基础：
        前额叶 Theta 波升高与抑郁、压抑情绪、皮质醇过高相关
        文献: PMC11770048 (MDD Theta 升高), PMC10813043 (HPA 轴过度活跃),
              PMC6970298 (前扣带回 Theta 异常与情绪调节)
        临床经验：缩放后平均波幅 > 13 µV → 压抑情绪/皮质醇过高风险
    """
    result = FeatureResult(
        name="情绪压力（Emotional Stress）",
        description="4-7 Hz Theta 频段分析",
        channel_a={},
        channel_b={},
        references=[
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11770048/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10813043/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6970298/",
        ],
    )

    # 年龄调整阈值
    threshold_raw_adj = get_age_adjusted_emotion_threshold(age)
    threshold_scaled_adj = round(threshold_raw_adj * THETA_EMOTION_SCALING, 3)

    for label, ch in [("A", ch_a), ("B", ch_b)]:
        freq_mask = (ch.freq_axis >= THETA_EMOTION_BAND[0]) & (ch.freq_axis <= THETA_EMOTION_BAND[1])

        band_amplitude_raw = np.mean(ch.spectrum[:, freq_mask], axis=1)

        segments = {}
        for seg_name, start, end in TIME_SEGMENTS:
            s = start
            e = end if end is not None else len(band_amplitude_raw)
            seg_amp = band_amplitude_raw[s:e]

            if len(seg_amp) == 0:
                continue

            seg_mean_raw = float(np.mean(seg_amp))
            exceed_count = int(np.sum(seg_amp > threshold_raw_adj))
            total = len(seg_amp)

            segments[seg_name] = {
                "method_a_threshold_raw": threshold_raw_adj,
                "method_a_threshold_scaled": threshold_scaled_adj,
                "method_a_mean_amplitude_raw": round(seg_mean_raw, 3),
                "method_a_mean_amplitude_scaled": round(seg_mean_raw * THETA_EMOTION_SCALING, 3),
                "method_a_mean_to_threshold_ratio": round(seg_mean_raw / threshold_raw_adj, 3),
                "method_a_exceed_count": exceed_count,
                "method_a_total_frames": total,
                "method_a_exceed_ratio": round(exceed_count / total, 4),
            }

        ch_result = {
            "threshold_raw": threshold_raw_adj,
            "threshold_scaled": threshold_scaled_adj,
            "segments": segments,
            "theta_amp_series": [round(float(x), 3) for x in band_amplitude_raw],
            "theta_drop_events": detect_theta_drop_events(
                band_amplitude_raw,
                baseline_limit=threshold_raw_adj * THETA_DROP_BASELINE_TOLERANCE,
            ),
        }
        events = ch_result["theta_drop_events"]
        ch_result["theta_drop_count"] = len(events)
        ch_result["theta_drop_total_duration"] = int(sum(e["duration"] for e in events))
        ch_result["theta_drop_max_pct"] = round(max((e["drop_pct"] for e in events), default=0.0), 3)

        if label == "A":
            result.channel_a = ch_result
        else:
            result.channel_b = ch_result

    a_ratio = result.channel_a["segments"]["full"]["method_a_exceed_ratio"]
    b_ratio = result.channel_b["segments"]["full"]["method_a_exceed_ratio"]
    a_mean_ratio = result.channel_a["segments"]["full"]["method_a_mean_to_threshold_ratio"]
    b_mean_ratio = result.channel_b["segments"]["full"]["method_a_mean_to_threshold_ratio"]
    a_drops = result.channel_a["theta_drop_count"]
    b_drops = result.channel_b["theta_drop_count"]
    max_ratio = max(a_ratio, b_ratio)
    max_drops = max(a_drops, b_drops)

    # 统一超标等级：超标占比 >10% 轻度 / >30% 明显 / >60% 严重；
    # Theta 骤降 4-8 次轻度 / ≥9 次明显，取两维度最高等级
    ratio_level = 3 if max_ratio > 0.6 else 2 if max_ratio > 0.3 else 1 if max_ratio > 0.1 else 0
    drop_level = 2 if max_drops >= 9 else 1 if max_drops >= 4 else 0
    overall_level = max(ratio_level, drop_level)

    # 等级来源说明
    level_sources = []
    if ratio_level > 0:
        pct_desc = "超过60%" if max_ratio > 0.6 else "超过30%" if max_ratio > 0.3 else "10-30%"
        level_sources.append(f"{pct_desc}的时间超过阈值")
    if drop_level > 0:
        drop_desc = "Theta 骤降≥9次" if max_drops >= 9 else "Theta 骤降4-8次"
        level_sources.append(drop_desc)

    parts = []
    if overall_level >= 3:
        parts.append(f"情绪压力信号严重（{'，'.join(level_sources)}），建议重点关注情绪状态")
    elif overall_level == 2:
        parts.append(f"情绪压力信号明显（{'，'.join(level_sources)}），建议关注情绪状态")
    elif overall_level == 1:
        parts.append(f"存在轻度情绪压力信号（{'，'.join(level_sources)}）")
    else:
        parts.append("情绪压力信号不明显（少于10%的时间超过阈值，Theta 骤降少于4次）")

    if max(a_mean_ratio, b_mean_ratio) > 2.0:
        parts.append("整体平均波幅达到阈值的两倍，持续高压状态")
    elif max(a_mean_ratio, b_mean_ratio) > 1.5:
        parts.append("整体平均波幅为阈值的1.5倍，情绪压力明显")
    elif max(a_mean_ratio, b_mean_ratio) > 1.0:
        parts.append("整体平均波幅超过阈值")

    # Theta 骤降分析（短期记忆影响）
    if max_drops >= 9:
        parts.append(f"Theta 波幅骤降次数明显（≥9次），提示压力激素已明显影响短期记忆功能，建议关注近期压力状态")
    elif max_drops >= 4:
        parts.append(f"Theta 波幅出现多次骤降（4-8次），提示压力激素可能已影响短期记忆功能，建议关注近期压力状态")
    elif max_drops >= 1:
        parts.append("Theta 波幅出现少量骤降（1-3次），属于正常波动范围")
    else:
        parts.append("Theta 波幅平稳，未见短期记忆受影响的迹象")

    result.interpretation = "；".join(parts)

    return result

def compute_eq_feature(ch_a: ChannelSpectrum,
                       ch_b: ChannelSpectrum,
                       age: int = 30) -> FeatureResult:
    """特征 #7: EQ 情商指数（Emotion Quotient）

    基于 Alpha 频段（8-12 Hz）左右脑不对称性分析。

    算法（对标 MATLAB EQ_main.m）:
        1. 提取闭眼期间 Alpha 频段平均幅值
        2. 计算逐帧左右脑差异 (A - B)
        3. 差值之和 > 0 → 情绪内收（direction 值 "Depress"）
           差值之和 ≤ 0 → 情绪外放（direction 值 "Brightness"）
           （direction 为内部标记，报告展示统一用中文）

    生理基础（Davidson Approach-Withdrawal 模型）:
        左前额叶更活跃（Alpha 更低）→ 正面情绪（外放）
        右前额叶更活跃（Alpha 更低）→ 负面情绪（压抑）
        Alpha 与皮层激活呈反比关系

    文献: PMC2835626 (Davidson), PMC3619694 (FAA 标准)
    """
    eq_result = compute_eq(
        ch_a.spectrum, ch_b.spectrum, ch_a.freq_axis,
        alpha_band=ALPHA_BAND,
        closed_eye_start=CLOSED_EYE_START,
        closed_eye_end=CLOSED_EYE_END,
        reduction_factor=EQ_REDUCTION_FACTOR,
    )

    result = FeatureResult(
        name="情绪指数（Emotion Quotient）",
        description=("闭眼期 Alpha(8-12Hz) 左右脑不对称性分析，反映情绪倾向。\n"
                     "图二「左右脑差异」中，向上的绿色条代表向内收的情绪，向下的蓝色条代表向外放的情绪；"
                     "差值总和反映整体倾向——数值偏正，说明情绪平均下来偏向内收；"
                     "数值偏负，说明情绪平均下来偏向外放。"),
        channel_a={},
        channel_b={},
        references=[
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC2835626/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC3619694/",
        ],
    )

    ce_frames = CLOSED_EYE_END - CLOSED_EYE_START
    ce_time = list(range(CLOSED_EYE_START + 1, CLOSED_EYE_END + 1))  # 显示用1基秒（41-80）

    result.channel_a = {
        "alpha_mean": round(float(np.mean(eq_result.A_alpha)), 3),
        "alpha_std": round(float(np.std(eq_result.A_alpha)), 3),
        "per_frame_diff": [round(float(v), 3) for v in eq_result.per_frame_diff],
        "ce_time": ce_time,
        "A_alpha": [round(float(v), 3) for v in eq_result.A_alpha],
        "B_alpha": [round(float(v), 3) for v in eq_result.B_alpha],
    }
    result.channel_b = {
        "alpha_mean": round(float(np.mean(eq_result.B_alpha)), 3),
        "alpha_std": round(float(np.std(eq_result.B_alpha)), 3),
        "per_frame_diff": [round(float(v), 3) for v in eq_result.per_frame_diff],
        "ce_time": ce_time,
        "A_alpha": [round(float(v), 3) for v in eq_result.A_alpha],
        "B_alpha": [round(float(v), 3) for v in eq_result.B_alpha],
    }

    diff_vals = [float(v) for v in eq_result.per_frame_diff]
    max_positive_bar = round(max((v for v in diff_vals if v > 0), default=0.0), 3)
    max_negative_bar = round(min((v for v in diff_vals if v < 0), default=0.0), 3)

    common = {
        "sum_diff": round(eq_result.sum_diff, 3),
        "direction": eq_result.direction,
        "positive_count": eq_result.positive_count,
        "negative_count": eq_result.negative_count,
        "reduction_factor": eq_result.reduction_factor,
        "total_frames": ce_frames,
        "max_positive_bar": max_positive_bar,
        "max_negative_bar": max_negative_bar,
    }
    result.channel_a.update(common)
    result.channel_b.update(common)

    dir_cn = "情绪内收" if eq_result.direction == "Depress" else "情绪外放"

    if eq_result.direction == "Depress":
        base_interpretation = (
            f"情绪倾向：{dir_cn}，"
            "情绪倾向向内收拢、较少外露，遇事更倾向先收在内心、较少主动向外表达"
        )
    else:
        base_interpretation = (
            f"情绪倾向：{dir_cn}，"
            "情绪容易向外流露与表达，遇事倾向积极面对、主动参与"
        )

    # 档位解读：压抑/外显侧各按强烈棒（≥6）条数取频次档文案（无强烈棒侧不显示），
    # 文案存于飞书建议库
    eq_patterns = _compute_eq_patterns(result.channel_a)
    tier_texts = _get_eq_interpretations(eq_patterns)
    if tier_texts:
        result.interpretation = base_interpretation + "；" + "；".join(tier_texts)
    else:
        result.interpretation = base_interpretation

    return result

def compute_paf_feature(ch_a: ChannelSpectrum,
                         ch_b: ChannelSpectrum,
                         age: int = 30,
                         filepath: str = None,
                         filter_mode: str = "per_second") -> FeatureResult:
    """特征 #8: PAF 峰值 Alpha 频率（Peak Alpha Frequency）

    使用 Tukey 窗 FFT（独立于 Hanning FFT），在闭眼期间逐帧检测峰值频率，
    然后通过年龄标定选择最终 PAF。

    算法（对标 MATLAB PAF_main.m + PAF_select.m）:
        1. Tukey 窗 FFT → 闭眼期间每帧 0-29 Hz 频谱
        2. 逐帧峰值检测（超过阈值）→ 按频率 bin 累计峰值计数
        3. BRQ 分析：3-15 Hz 范围内 findpeaks → 候选频率
        4. 年龄标定选择：在 age_std ± [-2, -1, 0, +1] 范围内选最高计数

    生理意义:
        PAF 反映大脑皮层成熟度和觉醒水平
        年龄越小 PAF 越低（儿童 5-8 Hz，成人 9-10 Hz，老年 7-8 Hz）
        PAF 偏低可能与注意力不集中、认知迟缓相关
    """
    if filepath:
        eeg = load_eeg_txt(filepath)
        if filter_mode == "global":
            A_raw = eeg.A_data.flatten()
            B_raw = eeg.B_data.flatten()
            A_proc = preprocess_signal(A_raw, eeg.sample_rate)
            B_proc = preprocess_signal(B_raw, eeg.sample_rate)
            A_frames = frame_signal(A_proc, eeg.sample_rate)
            B_frames = frame_signal(B_proc, eeg.sample_rate)
        else:
            A_frames = preprocess_frames(eeg.A_data, eeg.sample_rate)
            B_frames = preprocess_frames(eeg.B_data, eeg.sample_rate)
        A_spec_tukey, freq_axis_tukey = compute_fft(A_frames, PAF_WINDOW_TYPE, eeg.sample_rate)
        B_spec_tukey, _ = compute_fft(B_frames, PAF_WINDOW_TYPE, eeg.sample_rate)
    else:
        A_spec_tukey = ch_a.spectrum
        B_spec_tukey = ch_b.spectrum
        freq_axis_tukey = ch_a.freq_axis

    paf_result = compute_paf(
        A_spec_tukey, B_spec_tukey, freq_axis_tukey, age,
        closed_eye_start=CLOSED_EYE_START,
        closed_eye_end=CLOSED_EYE_END,
    )

    result = FeatureResult(
        name="Alpha峰值频率分析（Peak Alpha Frequency）",
        description="闭眼 Alpha 峰值频率分析，反映丘脑接收外来信息的效能；左丘脑以处理理性数理、逻辑数据为主，右丘脑以社交感性、创意信息为主。\n\n左右脑的峰值频率可能出现在不同的位置，两者组合起来会产生很多不同的模式与变量，因此这一部分只能给出相对笼统的解读与建议。不过，这组数据与一个人的性格和价值观密切相关；如需更有针对性的建议，需要先深入理解这份数据的解读方式，并咨询我们专家顾问的意见。",
        channel_a={},
        channel_b={},
        references=[],
        paf_result=paf_result,
    )

    brq_a_str = ", ".join(str(f) for f in paf_result.brq_peaks_a) if len(paf_result.brq_peaks_a) else "无"
    brq_b_str = ", ".join(str(f) for f in paf_result.brq_peaks_b) if len(paf_result.brq_peaks_b) else "无"

    result.channel_a = {
        "paf": paf_result.paf_a,
        "age_std": paf_result.age_std,
        "brq_peaks": brq_a_str,
        "peak_counts_3_15": [int(paf_result.peak_counts_a[f]) for f in range(3, 16)],
        "dom_ratio": paf_result.dom_ratio_a,
    }
    result.channel_b = {
        "paf": paf_result.paf_b,
        "age_std": paf_result.age_std,
        "brq_peaks": brq_b_str,
        "peak_counts_3_15": [int(paf_result.peak_counts_b[f]) for f in range(3, 16)],
        "dom_ratio": paf_result.dom_ratio_b,
    }

    # ── PAF 模式判定与解读（2026-10-13 改版）──────────────────
    # 模式（与飞书条件表失衡模式配套，解读文案全部从库里按模式取）：
    #   左脑快 / 右脑快：双侧均有稳定主峰（突出度≥30%）时才做快慢比较，
    #                    并在解读中说明信息处理侧重哪一边
    #   左脑不稳 / 右脑不稳：该侧主峰突出度 <30%，对应侧信息处理
    #                    速度可能在快与慢之间波动
    # 主峰偏离年龄标定值：按产品要求暂不显示任何解读与建议
    paf_patterns = _compute_paf_patterns(paf_result)
    tier_texts = _get_paf_interpretations(paf_patterns)

    parts = []
    has_a = "左脑不稳" not in paf_patterns
    has_b = "右脑不稳" not in paf_patterns

    if has_a and has_b and paf_result.paf_a == paf_result.paf_b:
        parts.append("左右脑均检出明显稳定的 PAF 主峰，且峰值频率一致，左右丘脑处理信息功能平衡稳定")
    else:
        if has_a and has_b:
            parts.append("左右脑均检出明显稳定的 PAF 主峰，左右丘脑处理信息功能稳定")
        if has_a:
            parts.append("左脑检出明显稳定的 PAF 主峰，左丘脑处理信息功能稳定")
        if has_b:
            parts.append("右脑检出明显稳定的 PAF 主峰，右丘脑处理信息功能稳定")

    parts.extend(tier_texts)

    # 四条模式（左脑快/右脑快/左脑不稳/右脑不稳）任一触发，
    # 解读末尾统一追加一次理想参考提示（避免多模式触发时重复）
    if paf_patterns:
        parts.append("较理想的 PAF 峰值状态可参考「脑波特征参考图例」第 6 图（稳定峰值频率）：左右脑峰值频率一致，且峰值明显、稳定")

    result.interpretation = "；".join(parts)

    return result

TBR_AGE_THRESHOLDS = {
    "child_7":     {"mean": 2.5, "elevated": 3.2, "clinical": 4.0},
    "child_7_11":  {"mean": 2.8, "elevated": 3.5, "clinical": 4.5},
    "teen_12_17":  {"mean": 2.4, "elevated": 3.1, "clinical": 3.8},
    "adult_18_25": {"mean": 2.3, "elevated": 3.0, "clinical": 3.7},
    "adult_26_40": {"mean": 2.1, "elevated": 2.7, "clinical": 3.3},
    "adult_41_60": {"mean": 1.9, "elevated": 2.5, "clinical": 3.1},
    "senior_60":   {"mean": 1.8, "elevated": 2.3, "clinical": 2.8},
}

def get_tbr_threshold(age: int) -> dict:
    """根据年龄返回 TBR 阈值。"""
    if age <= 7:
        return TBR_AGE_THRESHOLDS["child_7"]
    elif age <= 11:
        return TBR_AGE_THRESHOLDS["child_7_11"]
    elif age <= 17:
        return TBR_AGE_THRESHOLDS["teen_12_17"]
    elif age <= 25:
        return TBR_AGE_THRESHOLDS["adult_18_25"]
    elif age <= 40:
        return TBR_AGE_THRESHOLDS["adult_26_40"]
    elif age <= 60:
        return TBR_AGE_THRESHOLDS["adult_41_60"]
    else:
        return TBR_AGE_THRESHOLDS["senior_60"]

def compute_theta_beta_ratio(ch_a: ChannelSpectrum,
                             ch_b: ChannelSpectrum,
                             age: int = 30) -> FeatureResult:
    """特征 #2: Theta/Beta Ratio (TBR)

    Theta (4-8 Hz) / Beta (13-30 Hz) 功率比。
    TBR 升高与 ADHD、注意力不集中、皮层低觉醒相关。

    文献:
      - PMC8751092: TBR 年龄阈值
      - PMC9265331: TBR > 3.0 提示 ADHD（儿童青少年）
      - adhdbrainscan.co.uk: 各年龄段均值与标准差
      - PMC5135022: TBR 作为 ADHD 诊断辅助工具的实践指南

    注意:
      - 阈值基于 Cz/FCz 电极，前额 FP1/FP2 可能有差异
      - TBR 有较高假阳性率（>5%），不能单独用于诊断
      - 需结合其他指标综合判断
    """
    theta_band = (4, 8)
    beta_band = (13, 30)

    result = FeatureResult(
        name="Theta/Beta Ratio (TBR)",
        description="Theta(4-8Hz)/Beta(13-30Hz) 功率比，反映注意力和觉醒水平",
        channel_a={},
        channel_b={},
        references=[
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC8751092/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC9265331/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC5135022/",
        ],
    )

    thresholds = get_tbr_threshold(age)

    for label, ch in [("A", ch_a), ("B", ch_b)]:
        theta_mask = (ch.freq_axis >= theta_band[0]) & (ch.freq_axis < theta_band[1])
        beta_mask = (ch.freq_axis >= beta_band[0]) & (ch.freq_axis < beta_band[1])

        theta_power = np.sum(ch.spectrum[:, theta_mask] ** 2, axis=1)
        beta_power = np.sum(ch.spectrum[:, beta_mask] ** 2, axis=1)
        tbr_per_frame = theta_power / (beta_power + 1e-10)

        segments = {}
        for seg_name, start, end in TIME_SEGMENTS:
            s = start
            e = end if end is not None else len(tbr_per_frame)
            seg_tbr = tbr_per_frame[s:e]

            if len(seg_tbr) == 0:
                continue

            seg_mean = float(np.mean(seg_tbr))
            level = "normal"
            if seg_mean > thresholds["clinical"]:
                level = "clinical"
            elif seg_mean > thresholds["elevated"]:
                level = "elevated"

            segments[seg_name] = {
                "tbr_mean": round(seg_mean, 3),
                "tbr_std": round(float(np.std(seg_tbr)), 3),
                "tbr_max": round(float(np.max(seg_tbr)), 3),
                "level": level,
            }

        ch_result = {
            "thresholds": thresholds,
            "segments": segments,
        }

        if label == "A":
            result.channel_a = ch_result
        else:
            result.channel_b = ch_result

    a_tbr = result.channel_a["segments"]["full"]["tbr_mean"]
    b_tbr = result.channel_b["segments"]["full"]["tbr_mean"]
    a_level = result.channel_a["segments"]["full"]["level"]
    b_level = result.channel_b["segments"]["full"]["level"]

    level_map = {"normal": "正常范围", "elevated": "偏高", "clinical": "显著偏高"}
    parts = []
    parts.append(f"左前额叶 TBR={a_tbr:.2f}（{level_map[a_level]}）/ 右前额叶 TBR={b_tbr:.2f}（{level_map[b_level]}）")

    if max(a_tbr, b_tbr) > thresholds["clinical"]:
        parts.append("TBR 显著偏高，注意力状态值得关注")
    elif max(a_tbr, b_tbr) > thresholds["elevated"]:
        parts.append("TBR 偏高，建议关注注意力状态")
    else:
        parts.append("TBR 在正常范围内")

    parts.append(f"年龄阈值: 正常<{thresholds['mean']} / 偏高>{thresholds['elevated']} / 显著>{thresholds['clinical']}")

    result.interpretation = "；".join(parts)

    return result

def compute_delta_eyes_change(ch_a: ChannelSpectrum,
                               ch_b: ChannelSpectrum) -> FeatureResult:
    """特征 #3: Delta 相对功率睁闭眼变化

    对比睁眼 vs 闭眼时 Delta(0-3 Hz) 相对功率的变化。
    正常情况：闭眼时放松 → Delta 应下降或持平。
    异常情况：闭眼时 Delta 反而升高 → 提示闭眼时仍在紧张/焦虑。

    文献:
      - J Neurophysiol (2024): 压力导致前额 Delta 增加
      - PMC10719979: 正常成人额叶 Delta 相对功率基线
      - Front Psychol (2025): 睡眠剥夺后 Delta 升高与焦虑相关

    逻辑:
      delta_change = closed_ratio - open_ratio
      正值 = 闭眼时 Delta 升高（异常）
      负值 = 闭眼时 Delta 下降（正常放松）
    """
    delta_band = (0, 3)

    result = FeatureResult(
        name="Delta 睁闭眼占比变化",
        description="对比睁眼/闭眼 Delta(0-3Hz) 相对功率变化，评估闭眼时是否能有效放松",
        channel_a={},
        channel_b={},
        references=[
            "https://journals.physiology.org/doi/full/10.1152/jn.00618.2024",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10719979/",
            "https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1633875/full",
        ],
    )

    for label, ch in [("A", ch_a), ("B", ch_b)]:
        delta_mask = (ch.freq_axis >= delta_band[0]) & (ch.freq_axis <= delta_band[1])
        total_mask = ch.freq_axis <= MAX_DISPLAY_FREQ

        delta_power = np.sum(ch.spectrum[:, delta_mask] ** 2, axis=1)
        total_power = np.sum(ch.spectrum[:, total_mask] ** 2, axis=1)
        delta_ratio = delta_power / (total_power + 1e-10)

        # 三段分别计算
        seg_ratios = {}
        for seg_name, start, end in TIME_SEGMENTS:
            s = start
            e = end if end is not None else len(delta_ratio)
            seg = delta_ratio[s:e]
            if len(seg) == 0:
                continue
            seg_ratios[seg_name] = {
                "delta_ratio_mean": round(float(np.mean(seg)), 6),
                "delta_ratio_std": round(float(np.std(seg)), 6),
            }

        open1 = seg_ratios.get("eyes_open_1", {}).get("delta_ratio_mean", 0)
        closed = seg_ratios.get("eyes_closed", {}).get("delta_ratio_mean", 0)
        open2 = seg_ratios.get("eyes_open_2", {}).get("delta_ratio_mean", 0)

        change_1 = closed - open1
        change_2 = closed - open2

        ch_result = {
            "segments": seg_ratios,
            "change_closed_vs_open1": round(change_1, 6),
            "change_closed_vs_open2": round(change_2, 6),
        }

        if label == "A":
            result.channel_a = ch_result
        else:
            result.channel_b = ch_result

    a_change1 = result.channel_a["change_closed_vs_open1"]
    b_change1 = result.channel_b["change_closed_vs_open1"]

    parts = []
    if max(a_change1, b_change1) > 0.02:
        parts.append("闭眼时 Delta 占比反而升高，提示放松效果不理想")
    elif max(a_change1, b_change1) > 0:
        parts.append("闭眼时 Delta 占比略升，放松效果不理想")
    else:
        parts.append("闭眼时 Delta 占比下降，放松状态正常")

    result.interpretation = "；".join(parts)

    return result

THETA_SMR_THRESHOLDS = {
    "normal": 2.0,
    "elevated": 3.0,
    "clinical": 4.5,
}

def compute_theta_smr_ratio(ch_a: ChannelSpectrum,
                              ch_b: ChannelSpectrum) -> FeatureResult:
    """特征 #4: Theta/SMR Ratio

    Theta(4-8 Hz) / SMR(12-15 Hz) 功率比。
    SMR (Sensorimotor Rhythm) 对应感觉运动抑制和注意力集中。
    Theta/SMR 升高提示注意力不集中、走神、感觉运动抑制困难。

    文献:
      - ProZ QEEG: Theta/SMR > 3.0 提示无法安静坐下、入睡困难
      - Neurosity: SMR 神经反馈方案针对多动/冲动型
      - PMC7904968: SMR 与 TBR 是 ADHD 神经反馈两大标准方案
      - Semantic Scholar: 成功运动员有更低的 Theta/SMR

    注意:
      - SMR 传统在 C3/C4 测量最佳，FP1/FP2 信号较弱
      - 阈值中 3.0 有直接文献支持，其余为经验估算
    """
    theta_band = (4, 8)
    smr_band = (12, 15)

    result = FeatureResult(
        name="Theta/SMR Ratio",
        description="Theta(4-8Hz)/SMR(12-15Hz) 功率比，评估注意力和感觉运动抑制",
        channel_a={},
        channel_b={},
        references=["feature_04_05_theta_smr_delta_smr.md"],
    )

    thresholds = THETA_SMR_THRESHOLDS

    for label, ch in [("A", ch_a), ("B", ch_b)]:
        theta_mask = (ch.freq_axis >= theta_band[0]) & (ch.freq_axis < theta_band[1])
        smr_mask = (ch.freq_axis >= smr_band[0]) & (ch.freq_axis < smr_band[1])

        theta_power = np.sum(ch.spectrum[:, theta_mask] ** 2, axis=1)
        smr_power = np.sum(ch.spectrum[:, smr_mask] ** 2, axis=1)
        ratio_per_frame = theta_power / (smr_power + 1e-10)

        segments = {}
        for seg_name, start, end in TIME_SEGMENTS:
            s = start
            e = end if end is not None else len(ratio_per_frame)
            seg_ratio = ratio_per_frame[s:e]

            if len(seg_ratio) == 0:
                continue

            seg_mean = float(np.mean(seg_ratio))
            level = "normal"
            if seg_mean > thresholds["clinical"]:
                level = "clinical"
            elif seg_mean > thresholds["elevated"]:
                level = "elevated"

            segments[seg_name] = {
                "theta_smr_mean": round(seg_mean, 3),
                "theta_smr_std": round(float(np.std(seg_ratio)), 3),
                "theta_smr_max": round(float(np.max(seg_ratio)), 3),
                "level": level,
            }

        ch_result = {
            "thresholds": thresholds,
            "segments": segments,
        }

        if label == "A":
            result.channel_a = ch_result
        else:
            result.channel_b = ch_result

    a_val = result.channel_a["segments"]["full"]["theta_smr_mean"]
    b_val = result.channel_b["segments"]["full"]["theta_smr_mean"]
    a_level = result.channel_a["segments"]["full"]["level"]
    b_level = result.channel_b["segments"]["full"]["level"]

    level_map = {"normal": "正常范围", "elevated": "偏高", "clinical": "显著偏高"}
    parts = []
    parts.append(f"左前额叶 Theta/SMR={a_val:.2f}（{level_map[a_level]}）/ 右前额叶={b_val:.2f}（{level_map[b_level]}）")

    if max(a_val, b_val) > thresholds["clinical"]:
        parts.append("Theta/SMR 显著偏高，专注力状态值得关注")
    elif max(a_val, b_val) > thresholds["elevated"]:
        parts.append("Theta/SMR 偏高，建议关注注意力状态")
    else:
        parts.append("Theta/SMR 在正常范围内")

    parts.append(f"阈值: 正常<{thresholds['normal']} / 偏高>{thresholds['elevated']} / 显著>{thresholds['clinical']}")

    result.interpretation = "；".join(parts)

    return result

DELTA_SMR_THRESHOLDS = {
    "normal": 2.0,
    "elevated": 3.0,
    "clinical": 5.0,
}

def compute_delta_smr_ratio(ch_a: ChannelSpectrum,
                              ch_b: ChannelSpectrum) -> FeatureResult:
    """特征 #5: Delta/SMR Ratio

    Delta(0-3 Hz) / SMR(12-15 Hz) 功率比。
    Delta/SMR 升高提示白日梦、走神、严重低觉醒（接近困倦）。

    与 Theta/SMR 的区别:
      - Theta 主导 = 白日梦但未困倦（内聚焦状态）
      - Delta 主导 = 更严重低觉醒（出神/无意识状态）
      - ADD 更多表现为高 Delta，ADHD 更多表现为高 Theta

    文献:
      - Braboszcz & Delorme (2010): 走神时 Delta+Theta 升高
      - PMC8381606: 在线学习走神检测
      - Stress Therapy Solutions: ADD 表现为高 Delta 慢波
      - PMC11289613: ADHD 成人额叶 Delta 升高 + Beta 降低

    注意:
      - 阈值全部为经验估算，无直接文献支持
      - 需临床数据校准
    """
    delta_band = (0, 3)
    smr_band = (12, 15)

    result = FeatureResult(
        name="Delta/SMR Ratio",
        description="Delta(0-3Hz)/SMR(12-15Hz) 功率比，评估白日梦/走神和严重低觉醒",
        channel_a={},
        channel_b={},
        references=["feature_04_05_theta_smr_delta_smr.md"],
    )

    thresholds = DELTA_SMR_THRESHOLDS

    for label, ch in [("A", ch_a), ("B", ch_b)]:
        delta_mask = (ch.freq_axis >= delta_band[0]) & (ch.freq_axis <= delta_band[1])
        smr_mask = (ch.freq_axis >= smr_band[0]) & (ch.freq_axis < smr_band[1])

        delta_power = np.sum(ch.spectrum[:, delta_mask] ** 2, axis=1)
        smr_power = np.sum(ch.spectrum[:, smr_mask] ** 2, axis=1)
        ratio_per_frame = delta_power / (smr_power + 1e-10)

        segments = {}
        for seg_name, start, end in TIME_SEGMENTS:
            s = start
            e = end if end is not None else len(ratio_per_frame)
            seg_ratio = ratio_per_frame[s:e]

            if len(seg_ratio) == 0:
                continue

            seg_mean = float(np.mean(seg_ratio))
            level = "normal"
            if seg_mean > thresholds["clinical"]:
                level = "clinical"
            elif seg_mean > thresholds["elevated"]:
                level = "elevated"

            segments[seg_name] = {
                "delta_smr_mean": round(seg_mean, 3),
                "delta_smr_std": round(float(np.std(seg_ratio)), 3),
                "delta_smr_max": round(float(np.max(seg_ratio)), 3),
                "level": level,
            }

        ch_result = {
            "thresholds": thresholds,
            "segments": segments,
        }

        if label == "A":
            result.channel_a = ch_result
        else:
            result.channel_b = ch_result

    a_val = result.channel_a["segments"]["full"]["delta_smr_mean"]
    b_val = result.channel_b["segments"]["full"]["delta_smr_mean"]
    a_level = result.channel_a["segments"]["full"]["level"]
    b_level = result.channel_b["segments"]["full"]["level"]

    level_map = {"normal": "正常范围", "elevated": "偏高", "clinical": "显著偏高"}
    parts = []
    parts.append(f"左前额叶 Delta/SMR={a_val:.2f}（{level_map[a_level]}）/ 右前额叶={b_val:.2f}（{level_map[b_level]}）")

    if max(a_val, b_val) > thresholds["clinical"]:
        parts.append("Delta/SMR 显著偏高，觉醒状态值得关注")
    elif max(a_val, b_val) > thresholds["elevated"]:
        parts.append("Delta/SMR 偏高，建议关注觉醒状态")
    else:
        parts.append("Delta/SMR 在正常范围内")

    parts.append(f"阈值: 正常<{thresholds['normal']} / 偏高>{thresholds['elevated']} / 显著>{thresholds['clinical']}")

    result.interpretation = "；".join(parts)

    return result

def get_age_adjusted_tension_threshold(age: int) -> float:
    """根据年龄调整生理紧张绝对阈值（原始值）。

    基础阈值 56.3 µV（= 缩放后 20 µV / 0.355），对应成人。
    儿童和青少年的 Delta 基线更高（文献：4.5-19岁 Delta 相对功率
    显著高于成人），因此阈值需要相应上调。

    参考:
      - PMC8718919: Delta power 在 4.5-19 岁急剧下降，20 岁后稳定
      - PMC13076253: ADHD 儿童额叶 Delta 相对功率 ~29%
      - PMC8403030: 正常成人额叶 Delta 相对功率 ~10-24%

    调整规则（经验值，待临床数据校准）:
      age < 12:  阈值 × 1.5  (儿童 Delta 基线显著更高)
      12-18:     阈值 × 1.25 (青少年过渡期)
      19-60:     阈值 × 1.0  (成人基线)
      age > 60:  阈值 × 0.9  (老年人 Delta 略有回升)
    """
    base = TENSION_ABS_THRESHOLD_RAW
    if age < 12:
        return round(base * 1.5, 1)
    elif age < 19:
        return round(base * 1.25, 1)
    elif age <= 60:
        return base
    else:
        return round(base * 0.9, 1)

def get_age_adjusted_emotion_threshold(age: int) -> float:
    """根据年龄调整情绪压力绝对阈值（原始值）。

    基础阈值 54.2 µV（= 缩放后 13 µV / 0.24），对应成人。
    调整规则与生理紧张一致（儿童/青少年 Theta 基线更高）。

    调整规则（经验值，待临床数据校准）:
      age < 12:  阈值 × 1.5
      12-18:     阈值 × 1.25
      19-60:     阈值 × 1.0
      age > 60:  阈值 × 0.9
    """
    base = THETA_EMOTION_THRESHOLD_RAW
    if age < 12:
        return round(base * 1.5, 1)
    elif age < 19:
        return round(base * 1.25, 1)
    elif age <= 60:
        return base
    else:
        return round(base * 0.9, 1)

def parse_birthdate(text):
    """解析出生日期文本；支持 2010-05-20 / 2010/5/20 / 2010.5.20 / 2010年5月20日"""
    if not text or not text.strip():
        return None
    import re
    import datetime
    m = re.search(r'(\d{4})\s*[年./\-]\s*(\d{1,2})\s*[月./\-]\s*(\d{1,2})\s*日?', text)
    if not m:
        return None
    try:
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None

def parse_scan_date(filepath):
    """从数据文件头解析扫描开始日期。

    设备文件头格式：Wednesday. May 22.2024,17:28:10 Start Time
    兼容 "22 May 2024" 日前格式作为兜底。解析失败返回 None。"""
    import re
    import datetime
    months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
               "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
               "november": 11, "december": 12,
               "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
               "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                if "Start Time" not in line:
                    continue
                m = re.search(r'([A-Za-z]+)[.\s]+(\d{1,2})[.\s,-]+(\d{4})', line)
                mon = months.get(m.group(1).lower()) if m else None
                if mon:
                    try:
                        return datetime.date(int(m.group(3)), mon, int(m.group(2)))
                    except ValueError:
                        return None
                m = re.search(r'(\d{1,2})\s+([A-Za-z]+)[.\s,]+\s*(\d{4})', line)
                mon = months.get(m.group(2).lower()) if m else None
                if mon:
                    try:
                        return datetime.date(int(m.group(3)), mon, int(m.group(1)))
                    except ValueError:
                        return None
                break
    except Exception:
        pass
    return None

def compute_age_from_birthdate(birth, ref_date):
    """按参考日期（扫描日）计算周岁年龄：当年生日未到减一岁"""
    age = ref_date.year - birth.year
    if (ref_date.month, ref_date.day) < (birth.month, birth.day):
        age -= 1
    return age

def load_and_process(filepath: str, filter_mode: str = "per_second") -> tuple:
    """Load TXT, preprocess, compute FFT for both channels.

    Args:
        filepath: Path to EEG TXT file
        filter_mode: "per_second" (逐秒独立滤波, matching MATLAB) or "global" (全局连续滤波)

    Returns: (ch_a_spectrum, ch_b_spectrum, n_seconds)
    """
    print(f"\nProcessing: {filepath}")
    eeg = load_eeg_txt(filepath)
    print(f"  Sample rate: {eeg.sample_rate} Hz")
    print(f"  Duration: {eeg.n_seconds} seconds")
    print(f"  Channels: A ({eeg.A_data.shape[0]} frames), B ({eeg.B_data.shape[0]} frames)")
    print(f"  Filter mode: {filter_mode}")

    if filter_mode == "global":
        A_raw = eeg.A_data.flatten()
        B_raw = eeg.B_data.flatten()
        A_proc = preprocess_signal(A_raw, eeg.sample_rate)
        B_proc = preprocess_signal(B_raw, eeg.sample_rate)
        A_frames = frame_signal(A_proc, eeg.sample_rate)
        B_frames = frame_signal(B_proc, eeg.sample_rate)
    else:
        A_frames = preprocess_frames(eeg.A_data, eeg.sample_rate)
        B_frames = preprocess_frames(eeg.B_data, eeg.sample_rate)

    A_spectrum, freq_axis = compute_fft(A_frames, "hanning", eeg.sample_rate)
    B_spectrum, _ = compute_fft(B_frames, "hanning", eeg.sample_rate)

    time_axis = np.arange(A_spectrum.shape[0])

    # Apply scaling factors (官方增益表，用于3D可视化)
    # 2026-11-04 起对标原版软件，弃用 V2 表，改用官方 V10 表
    scaling = get_scaling_array(freq_axis)
    A_spectrum_scaled = A_spectrum * scaling[np.newaxis, :]
    B_spectrum_scaled = B_spectrum * scaling[np.newaxis, :]

    # Clip to 0-30 Hz for display
    mask = freq_axis <= MAX_DISPLAY_FREQ
    ch_a = ChannelSpectrum(
        spectrum=A_spectrum[:, mask],
        freq_axis=freq_axis[mask],
        time_axis=time_axis,
        spectrum_scaled=A_spectrum_scaled[:, mask],
    )
    ch_b = ChannelSpectrum(
        spectrum=B_spectrum[:, mask],
        freq_axis=freq_axis[mask],
        time_axis=time_axis,
        spectrum_scaled=B_spectrum_scaled[:, mask],
    )

    return ch_a, ch_b, eeg.n_seconds

def load_and_process_dual(filepath: str) -> tuple:
    """
    加载并处理EEG数据，同时生成两套频谱数据：
    - 全局连续滤波：用于生理紧张、情绪压力、情绪指数等（低频更准确）
    - 逐秒独立滤波：仅用于 PAF 计算（保留时间分辨率）

    Returns: (ch_a_global, ch_b_global, ch_a_persecond, ch_b_persecond, n_seconds)
    """
    print(f"\nProcessing: {filepath}")
    eeg = load_eeg_txt(filepath)
    print(f"  Sample rate: {eeg.sample_rate} Hz")
    print(f"  Duration: {eeg.n_seconds} seconds")
    print(f"  Channels: A ({eeg.A_data.shape[0]} frames), B ({eeg.B_data.shape[0]} frames)")
    print(f"  Filter mode: dual (全局连续滤波 + 逐秒独立滤波)")

    # === 全局连续滤波 ===
    A_raw = eeg.A_data.flatten()
    B_raw = eeg.B_data.flatten()
    A_proc_global = preprocess_signal(A_raw, eeg.sample_rate)
    B_proc_global = preprocess_signal(B_raw, eeg.sample_rate)
    A_frames_global = frame_signal(A_proc_global, eeg.sample_rate)
    B_frames_global = frame_signal(B_proc_global, eeg.sample_rate)

    A_spec_global, freq_axis = compute_fft(A_frames_global, "hanning", eeg.sample_rate)
    B_spec_global, _ = compute_fft(B_frames_global, "hanning", eeg.sample_rate)

    # === 逐秒独立滤波 ===
    A_frames_per = preprocess_frames(eeg.A_data, eeg.sample_rate)
    B_frames_per = preprocess_frames(eeg.B_data, eeg.sample_rate)

    A_spec_per, _ = compute_fft(A_frames_per, "hanning", eeg.sample_rate)
    B_spec_per, _ = compute_fft(B_frames_per, "hanning", eeg.sample_rate)

    time_axis = np.arange(A_spec_global.shape[0])

    # 缩放因子（官方增益表，两套数据共享同一频率轴，用于3D可视化）
    # 2026-11-04 起对标原版软件，弃用 V2 表，改用官方 V10 表
    scaling = get_scaling_array(freq_axis)

    # 裁剪到 0-30 Hz
    mask = freq_axis <= MAX_DISPLAY_FREQ

    # 构建全局滤波的 ChannelSpectrum
    ch_a_global = ChannelSpectrum(
        spectrum=A_spec_global[:, mask],
        freq_axis=freq_axis[mask],
        time_axis=time_axis,
        spectrum_scaled=A_spec_global[:, mask] * scaling[np.newaxis, :][:, mask],
    )
    ch_b_global = ChannelSpectrum(
        spectrum=B_spec_global[:, mask],
        freq_axis=freq_axis[mask],
        time_axis=time_axis,
        spectrum_scaled=B_spec_global[:, mask] * scaling[np.newaxis, :][:, mask],
    )

    # 构建逐秒滤波的 ChannelSpectrum
    ch_a_persecond = ChannelSpectrum(
        spectrum=A_spec_per[:, mask],
        freq_axis=freq_axis[mask],
        time_axis=time_axis,
        spectrum_scaled=A_spec_per[:, mask] * scaling[np.newaxis, :][:, mask],
    )
    ch_b_persecond = ChannelSpectrum(
        spectrum=B_spec_per[:, mask],
        freq_axis=freq_axis[mask],
        time_axis=time_axis,
        spectrum_scaled=B_spec_per[:, mask] * scaling[np.newaxis, :][:, mask],
    )

    return ch_a_global, ch_b_global, ch_a_persecond, ch_b_persecond, eeg.n_seconds

def _compute_dsr_patterns(relaxation, attention, concentration, for_advice=False):
    """根据放松力/注意力/集中力三项得分计算已触发的失衡模式（与报告解读逻辑一一对应）：
    单项低于 20 为偏低；项间相差 ≥ 5 为失衡；三项均衡（无偏低、相差 ≤4）但有单项
    高于 30 时触发"三项整体偏高"（仅解读展示，不给具体建议）。
    for_advice=True 时执行方向修正：失衡对的高项 > 30 时，失衡主因是高项绝对偏高
    而非低项偏低（低项 ≥20 即在健康区间），抑制该对失衡模式（避免"提升低项"建议），
    改挂"X偏高伴随失衡"标记（触发压低高项建议）；低项 < 20 的偏低模式不受影响，
    高项>30 且低项<20 时压低+提升两方向并存。"""
    patterns = []
    if relaxation < 20:
        patterns.append("放松力偏低")
    if attention < 20:
        patterns.append("注意力偏低")
    if concentration < 20:
        patterns.append("集中力偏低")
    if concentration >= relaxation + 5:
        patterns.append("集中力高于放松力")
    if concentration >= attention + 5:
        patterns.append("集中力高于注意力")
    if concentration <= relaxation - 5:
        patterns.append("集中力低于放松力")
    if concentration <= attention - 5:
        patterns.append("集中力低于注意力")
    if attention >= relaxation + 5:
        patterns.append("注意力高于放松力")
    if attention <= relaxation - 5:
        patterns.append("注意力低于放松力")
    if not patterns and max(relaxation, attention, concentration) > 30:
        patterns.append("三项整体偏高")
    if for_advice:
        # 2026-10-20 方向修正通用化（放松力/注意力/集中力任一作为高项均适用）：
        # 高项超出健康上限（>30）时抑制该对失衡模式，干预方向由"X偏高伴随失衡"
        # 标记承载（压低高项建议）。双高对（两项均 >30）只压高项，低项自身是否
        # 处理看它与第三项的失衡。"三项整体偏高"（无项间失衡）仅给解读不挂标记。
        _high_pairs = [
            ("集中力高于放松力", concentration, "集中力偏高伴随失衡"),
            ("集中力高于注意力", concentration, "集中力偏高伴随失衡"),
            ("集中力低于放松力", relaxation, "放松力偏高伴随失衡"),
            ("集中力低于注意力", attention, "注意力偏高伴随失衡"),
            ("注意力高于放松力", attention, "注意力偏高伴随失衡"),
            ("注意力低于放松力", relaxation, "放松力偏高伴随失衡"),
        ]
        _markers = []
        for _pat, _high, _marker in _high_pairs:
            if _pat in patterns and _high > 30:
                patterns.remove(_pat)
                if _marker not in _markers:
                    _markers.append(_marker)
        patterns.extend(_markers)
    return patterns

DSR_PATTERN_ORDER = [
    "放松力偏低", "注意力偏低", "集中力偏低",
    "集中力高于放松力", "集中力高于注意力",
    "集中力低于放松力", "集中力低于注意力",
    "注意力高于放松力", "注意力低于放松力",
    "三项整体偏高",
]

def _get_dsr_interpretations(patterns):
    """从建议库按失衡模式取动态自我调节解读文案，按固定展示顺序返回"""
    if not patterns:
        return []
    lib = _get_advice_library()
    if lib is None:
        return []
    texts = {}
    for t in lib.thresholds:
        if (t.feature and t.feature.code == "self_regulation"
                and t.condition_type == "分级阈值"
                and t.pattern and t.interpretation):
            for p in t.pattern:
                if p in patterns:
                    texts[p] = t.interpretation
    return [texts[p] for p in DSR_PATTERN_ORDER if p in texts]

def _compute_eq_patterns(ch):
    """根据情绪指数棒形（per_frame_diff）计算已触发的档位模式。

    2026-10-13 频次分档改版：
    - 强烈棒 = 同侧棒值绝对值 ≥6 的棒，按条数分频次档
      （压抑/外显侧独立计数、独立显示）：
       1-3 条 = 偶尔、4-6 条 = 较多、7 条以上 = 频繁
    - 轻微/轻度/中度深度档暂停使用（按产品要求暂不出档位解读，
      原"按最深棒分档"逻辑保留在下方注释中，可随时恢复）
    总值偏正或压抑帧多于外显帧时，附加"压抑倾向"
    （接纳情绪类建议的触发模式；「压抑侧最深棒更深」判定 2026-10-20 暂停，见下方注释）。
    """
    diff = ch.get("per_frame_diff") or []
    if not diff:
        return []
    patterns = []
    strong_pos = sum(1 for v in diff if v >= 6)    # 压抑方向强烈棒
    strong_neg = sum(1 for v in diff if v <= -6)   # 外显方向强烈棒
    if strong_pos:
        if strong_pos >= 7:
            patterns.append("频繁强烈压抑")
        elif strong_pos >= 4:
            patterns.append("较多强烈压抑")
        else:
            patterns.append("偶尔强烈压抑")
    if strong_neg:
        if strong_neg >= 7:
            patterns.append("频繁强烈外显")
        elif strong_neg >= 4:
            patterns.append("较多强烈外显")
        else:
            patterns.append("偶尔强烈外显")
    # —— 暂停使用的深度分档（2026-10-13 暂停，逻辑保留备恢复）——
    # pos_bars = [v for v in diff if v > 0]
    # neg_bars = [v for v in diff if v < 0]
    # if pos_bars:
    #     depth = max(pos_bars)
    #     if depth > 6: patterns.append("压抑剧烈")
    #     elif depth > 4: patterns.append("压抑中度")
    #     elif depth > 2: patterns.append("压抑轻度")
    #     else: patterns.append("压抑轻微")
    # if neg_bars:
    #     depth = min(neg_bars)
    #     if depth < -6: patterns.append("外显剧烈")
    #     elif depth < -4: patterns.append("外显中度")
    #     elif depth < -2: patterns.append("外显轻度")
    #     else: patterns.append("外显轻微")
    sum_diff = ch.get("sum_diff", 0) or 0
    pos_cnt = ch.get("positive_count", 0) or 0
    neg_cnt = ch.get("negative_count", 0) or 0
    # 2026-10-20 暂停「压抑侧最深棒 > 外显侧最深棒」判定：单帧最深棒略深即触发
    # （陈祥榕案例 5.03 vs 4.50 仅差 0.53），与整体外放方向矛盾，外放者拿到压抑向建议
    # max_pos = max((v for v in diff if v > 0), default=0.0)
    # max_neg_abs = abs(min((v for v in diff if v < 0), default=0.0))
    if sum_diff > 0 or pos_cnt > neg_cnt:
        patterns.append("压抑倾向")
    return patterns

EQ_PATTERN_ORDER = [
    "偶尔强烈压抑", "较多强烈压抑", "频繁强烈压抑",
    "偶尔强烈外显", "较多强烈外显", "频繁强烈外显",
]

def _get_eq_interpretations(patterns):
    """从建议库按档位模式取情绪指数解读文案，按固定展示顺序返回"""
    if not patterns:
        return []
    lib = _get_advice_library()
    if lib is None:
        return []
    texts = {}
    for t in lib.thresholds:
        if (t.feature and t.feature.code == "emotional_index"
                and t.condition_type == "分级阈值"
                and t.pattern and t.interpretation):
            for p in t.pattern:
                if p in patterns:
                    texts[p] = t.interpretation
    return [texts[p] for p in EQ_PATTERN_ORDER if p in texts]

def _compute_paf_patterns(paf_result):
    """根据 PAF 计算结果判定已触发的模式。

    规则（2026-10-13 调整）：
    - 左脑不稳 / 右脑不稳：该侧主峰突出度 < 30%（或未检出主峰）
    - 左脑快 / 右脑快：左右 PAF 有差异即触发，与主峰是否明显无关
      （快的一侧触发对应模式，可与不稳模式并存）
    - 主峰偏离年龄标定值不产生任何模式（按产品要求暂不显示）
    """
    has_a = (paf_result.dom_ratio_a is not None
             and paf_result.dom_ratio_a >= PAF_PROM_SIGNIFICANT)
    has_b = (paf_result.dom_ratio_b is not None
             and paf_result.dom_ratio_b >= PAF_PROM_SIGNIFICANT)

    patterns = []
    if not has_a:
        patterns.append("左脑不稳")
    if not has_b:
        patterns.append("右脑不稳")
    if paf_result.paf_a != paf_result.paf_b:
        patterns.append("左脑快" if paf_result.paf_a > paf_result.paf_b else "右脑快")
    return patterns

PAF_PATTERN_ORDER = ["左脑快", "右脑快", "左脑不稳", "右脑不稳"]

def _compute_paf_patterns_from_channels(ch_a, ch_b):
    """从通道字典（channel_a/b）判定 PAF 模式，规则同 _compute_paf_patterns。

    供超标等级判定等只有通道字典、没有 paf_result 的调用方使用。
    """
    dom_a = ch_a.get("dom_ratio")
    dom_b = (ch_b or {}).get("dom_ratio")
    has_a = dom_a is not None and dom_a >= PAF_PROM_SIGNIFICANT
    has_b = dom_b is not None and dom_b >= PAF_PROM_SIGNIFICANT

    patterns = []
    if not has_a:
        patterns.append("左脑不稳")
    if not has_b:
        patterns.append("右脑不稳")
    paf_a = ch_a.get("paf")
    paf_b = (ch_b or {}).get("paf")
    if paf_a is not None and paf_b is not None and paf_a != paf_b:
        patterns.append("左脑快" if paf_a > paf_b else "右脑快")
    return patterns

def _get_paf_interpretations(patterns):
    """从建议库按模式取 PAF 解读文案，按固定展示顺序返回"""
    if not patterns:
        return []
    lib = _get_advice_library()
    if lib is None:
        return []
    texts = {}
    for t in lib.thresholds:
        if (t.feature and t.feature.code == "paf"
                and t.condition_type == "分级阈值"
                and t.pattern and t.interpretation):
            for p in t.pattern:
                if p in patterns:
                    texts[p] = t.interpretation
    return [texts[p] for p in PAF_PATTERN_ORDER if p in texts]
