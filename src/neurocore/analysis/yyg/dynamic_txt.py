# -*- coding: utf-8 -*-
"""
动态扫描 TXT → 放松力 / 注意力 / 集中力百分比

移植自 YYG_datacollection/fft-live/dynamic-core.js 的 MonitorR N=128 已验证算法，
并遵循《DYNAMIC-TXT-REPORT-AI-HANDOFF》（核对日期 2026-11-09）实现约定。

口径（2026-11-09 用户确认）：
- TXT 为自研工具导出的三轮合并格式（128 点/块，与官方结构一致，共 183 块）；
- 三轮切分：放松(Alpha, mode=1) [0,61) → 注意(SMR, mode=4) [61,122)
  → 集中(Beta, mode=2) [122,183)，每轮 60 个显示格 + 1 个边界计算块，
  边界块只更新阈值历史，不计入成绩分子与分母；
- 每轮阈值冷启动（三轮阈值互不影响）；
- 128 点 TXT 未保存完整 256 槽门控缓冲，门控按“通过”假设，结果为估算；
- percentage = floor(100 × 本轮成功显示格数 / 60)，向下取整，不四舍五入。

无第三方依赖。数值链路与 JS 参考实现逐位对齐：周期 Hann 窗、
FFT N=128（前向、无 1/N 归一化）、p[k]=0.5·|X[k]|、p[0]=0、
七项频谱指标、三次分组历史自适应阈值（每块先更新阈值再严格比较）。
"""
import math
import re

N = 128                       # FFT 点数 = 每块每通道采样点数
VISIBLE_CELLS = 60            # 每轮显示格数（计划分钟数 × 60）
BLOCKS_PER_ROUND = 61         # 每轮计算块 = 60 显示格 + 1 边界块
ROUND_MODES = (1, 4, 2)       # 放松 → 注意 → 集中
MODES = {1: "Alpha · 放松", 4: "SMR · 注意力", 2: "Beta · 集中力"}
ROUND_LABELS = {1: "放松力", 4: "注意力", 2: "集中力"}
ALGORITHM = "monitorr-n128-v1"
SPREAD_COEF = 0.03167420814479638

# 七项指标：前三项达标方向“大于阈值”，UPPER 集合内为“小于阈值”
KEYS = ("smr_B", "theta_AB", "high_frequency_AB", "beta_A",
        "relax_spectral_support", "relax_auxiliary", "beta_inhibit_AB")
UPPER = frozenset(("theta_AB", "high_frequency_AB", "relax_auxiliary", "beta_inhibit_AB"))
_REQUIRED = {1: ("relax_spectral_support", "relax_auxiliary"),
             4: ("smr_B", "theta_AB", "high_frequency_AB"),
             2: ("beta_A", "theta_AB", "beta_inhibit_AB")}


class DynamicTXTParseError(ValueError):
    """动态 TXT 结构不正确，或无法按已确认口径计算。"""


# ── 频谱（对齐 dynamic-core.js spectrum）─────────────────────

_HANN = tuple(0.5 - 0.5 * math.cos(2 * math.pi * i / N) for i in range(N))


def _validate_samples(samples):
    if len(samples) != N:
        raise DynamicTXTParseError(f"每个数据块必须恰好 {N} 个整数，当前 {len(samples)} 个")
    for x in samples:
        if x < 0 or x > 1023:
            raise DynamicTXTParseError("采样码必须为 0–1023 的整数")


def spectrum(raw):
    """128 点块 → p[0..63]：p[k] = 0.5·|X[k]|，p[0]=0。"""
    _validate_samples(raw)
    re_part = [(raw[i] - 511) * 50 / 512 * _HANN[i] for i in range(N)]
    im_part = [0.0] * N
    # 位反转置换（与 JS 实现相同的循环写法）
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            re_part[i], re_part[j] = re_part[j], re_part[i]
    # 原地基-2 蝶形
    length = 2
    while length <= N:
        angle = -2 * math.pi / length
        half = length // 2
        for start in range(0, N, length):
            for k in range(half):
                c = math.cos(angle * k)
                s = math.sin(angle * k)
                i1 = start + k
                i2 = i1 + half
                tr = re_part[i2] * c - im_part[i2] * s
                ti = re_part[i2] * s + im_part[i2] * c
                re_part[i2] = re_part[i1] - tr
                im_part[i2] = im_part[i1] - ti
                re_part[i1] += tr
                im_part[i1] += ti
        length *= 2
    out = [math.hypot(re_part[k], im_part[k]) * 0.5 for k in range(64)]
    out[0] = 0.0
    return out


def _mean(a, lo, hi):
    s = 0.0
    for i in range(lo, hi):
        s += a[i]
    return s / (hi - lo)


def _spread(p):
    # 求和含 26 个 bin（k=4..29），centre 分母保持 25：既定实现，非笔误
    s = 0.0
    for i in range(4, 30):
        s += p[i]
    centre = s / 25
    squares = 0.0
    for i in range(4, 30):
        squares += (p[i] - centre) ** 2
    return math.sqrt(squares) * 0.2


def compute_metrics(a, b):
    """七项频谱指标。beta_inhibit_AB 与 relax_spectral_support 数值相同，
    但各自维护独立阈值状态（与 JS 一致）。"""
    pa = spectrum(a)
    pb = spectrum(b)
    support = (_mean(pa, 9, 19) + _mean(pb, 9, 19)) * 0.5
    return {
        "smr_B": _mean(pb, 12, 16),
        "theta_AB": (_mean(pa, 4, 8) + _mean(pb, 4, 8)) * 0.5,
        "high_frequency_AB": (_mean(pa, 19, 49) + _mean(pb, 19, 49)) * 0.5,
        "beta_A": _mean(pa, 16, 19),
        "relax_spectral_support": support,
        "beta_inhibit_AB": support,
        "relax_auxiliary": (math.log(0.94 + SPREAD_COEF * _spread(pa))
                            + math.log(0.94 + SPREAD_COEF * _spread(pb))) * 50,
    }


# ── 自适应阈值与奖励引擎 ──────────────────────────────────────

class AdaptiveThreshold:
    """三次分组历史阈值：冷启动 counter=0、history=[0,0,0]、value=0；
    第 4 次起 value=(均值+极值)/2 并重置计数器；比较严格大于/小于。"""

    def __init__(self, upper):
        self.upper = upper
        self.counter = 0
        self.history = [0.0, 0.0, 0.0]
        self.value = 0.0

    def update(self, x):
        if not math.isfinite(x):
            raise ValueError("指标不是有限数值")
        if self.counter < 3:
            self.history[self.counter] = x
            self.counter += 1
        else:
            m = (self.history[0] + self.history[1] + self.history[2]) / 3
            extreme = max(self.history) if self.upper else min(self.history)
            self.value = (m + extreme) / 2
            self.counter = 1
            self.history[0] = x
        return self.value


def combine(mode, flags):
    """模式组合：同块条件全部满足才为候选（enable 默认为 1）。"""
    required = _REQUIRED.get(mode)
    if required is None:
        raise ValueError("只支持 Alpha、SMR、Beta 三种模式")
    return all(flags[k] is True for k in required)


class DynamicEngine:
    """七项指标各自独立阈值；每块先更新全部阈值，再用更新后的值判定本块。"""

    def __init__(self):
        self.thresholds = {k: AdaptiveThreshold(k in UPPER) for k in KEYS}
        self.frames = 0

    def process(self, a, b, mode, gate=True):
        if mode not in MODES:
            raise ValueError("不支持的训练模式")
        values = compute_metrics(a, b)
        flags = {}
        thresholds = {}
        for k in KEYS:
            t = self.thresholds[k]
            thresholds[k] = t.update(values[k])
            flags[k] = values[k] < t.value if t.upper else values[k] > t.value
        candidate = combine(mode, flags)
        self.frames += 1
        if not candidate:
            reward = False
        elif gate is None:
            reward = None
        else:
            reward = gate
        return {"values": values, "thresholds": thresholds, "flags": flags,
                "candidate": candidate, "reward": reward}


# ── TXT 解析（128 点/块官方结构）────────────────────────────

_BLOCK_RE = re.compile(r"^([AB])\s+(\d+)$")
_ROW_RE = re.compile(r"^\s*\d+(\s*,\s*\d+)*\s*$")


def _header_number(line):
    raw = line.split(":", 1)[1].strip()
    try:
        return int(raw)
    except ValueError:
        raise DynamicTXTParseError(f"文件头数值无法识别：{line.strip()}")


def parse_dynamic_txt(text):
    """解析三轮合并动态 TXT；允许 UTF-8 BOM、CRLF/LF。
    保留全部实际数据块，文件头申报数不一致仅警告、不截断。"""
    if text.startswith("\ufeff"):
        text = text[1:]
    lines = text.splitlines()
    n_line = next((l for l in lines if re.match(r"^Sample Per Line:", l)), None)
    if n_line is not None and _header_number(n_line) == 256:
        raise DynamicTXTParseError(
            "该 TXT 为 256 点/行格式（静态扫描或旧版原始导出），"
            "本功能只支持 128 点/块的三轮动态扫描 TXT")
    if n_line is None or _header_number(n_line) != N:
        raise DynamicTXTParseError(
            "请导入 128 点/块的三轮动态扫描 TXT（缺少 Sample Per Line 头或数值不是 128）")
    channel_line = next((l for l in lines if re.match(r"^Channel Number:", l)), None)
    if channel_line is None or _header_number(channel_line) != 2:
        raise DynamicTXTParseError("文件必须包含两个通道（Channel Number: 2）")
    total_line = next((l for l in lines if re.match(r"^Total Sample:", l)), None)
    declared = _header_number(total_line) if total_line is not None else None

    blocks = {"A": [], "B": []}
    offsets = {"A": [], "B": []}
    i = 0
    while i < len(lines):
        m = _BLOCK_RE.match(lines[i].strip())
        if m is None:
            i += 1
            continue
        i += 1
        if i >= len(lines):
            raise DynamicTXTParseError("原始数据行缺失")
        row = lines[i]
        if _ROW_RE.match(row) is None:
            raise DynamicTXTParseError("原始数据行不正确")
        samples = [int(t) for t in row.split(",")]
        _validate_samples(samples)
        ch = m.group(1)
        blocks[ch].append(samples)
        offsets[ch].append(int(m.group(2)))
        i += 1

    if not blocks["A"] or len(blocks["A"]) != len(blocks["B"]):
        raise DynamicTXTParseError("A/B 数据块缺失或数量不一致")
    for idx in range(len(blocks["A"])):
        if offsets["A"][idx] != idx * N or offsets["B"][idx] != idx * N:
            raise DynamicTXTParseError("A/B 块偏移不连续或不一致")
    frames = list(zip(blocks["A"], blocks["B"]))
    actual = len(frames) * N
    warnings = []
    if declared is not None and declared != actual:
        warnings.append(
            f"文件头 Total Sample 申报 {declared}，正文实际每通道 {actual}；已保留全部实际数据块")
    return {"frames": frames, "declared_total": declared,
            "actual_samples": actual, "warnings": warnings}


# ── 轮次计算与对外入口 ───────────────────────────────────────

def compute_round(frames, mode, engine=None, seconds=VISIBLE_CELLS):
    """处理一轮（至多 seconds+1 块）：前 seconds 块计成绩，
    末尾边界块只更新阈值。门控按“通过”假设，结果为估算。"""
    if mode not in MODES:
        raise DynamicTXTParseError("需要明确的训练模式（1 放松 / 4 注意 / 2 集中）")
    if engine is None:
        engine = DynamicEngine()
    if len(frames) > seconds + 1:
        raise DynamicTXTParseError(
            f"本轮数据块 {len(frames)} 个，超过计划 {seconds} 秒 + 1 边界块")
    success = 0
    for index, (a, b) in enumerate(frames):
        result = engine.process(a, b, mode, gate=True)
        if index < seconds and result["reward"] is True:
            success += 1
    return {
        "mode": mode,
        "name": MODES[mode],
        "label": ROUND_LABELS[mode],
        "seconds": seconds,
        "visible_cells": seconds,
        "observed_visible_cells": min(len(frames), seconds),
        "boundary_blocks": 1 if len(frames) > seconds else 0,
        "success_cells": success,
        "percentage": 100 * success // seconds,
        "status": "partial" if len(frames) < seconds + 1 else "estimated",
    }


def compute_from_txt(text):
    """三轮合并动态 TXT → 放松力/注意力/集中力百分比（估算）。"""
    dataset = parse_dynamic_txt(text)
    frames = dataset["frames"]
    expected = BLOCKS_PER_ROUND * len(ROUND_MODES)
    if len(frames) != expected:
        raise DynamicTXTParseError(
            f"动态扫描 TXT 应包含三轮共 {expected} 块"
            f"（放松/注意/集中各 60 秒 + 1 边界块），当前文件实际 {len(frames)} 块；"
            "仅凭 TXT 无法确定轮次边界，请提供完整三轮合并 TXT")
    rounds = [compute_round(frames[i * BLOCKS_PER_ROUND:(i + 1) * BLOCKS_PER_ROUND], mode)
              for i, mode in enumerate(ROUND_MODES)]
    return {
        "algorithm": ALGORITHM,
        "source": "txt-recompute",
        "status": "estimated",
        "relaxation": rounds[0]["percentage"],
        "attention": rounds[1]["percentage"],
        "concentration": rounds[2]["percentage"],
        "assumptions": [
            "三轮切分：放松→注意→集中，每轮 60 显示格 + 1 边界块",
            "每轮阈值冷启动（三轮互不影响）",
            "原始门控按通过假设（128 点 TXT 未保存完整门控缓冲）",
        ],
        "warnings": dataset["warnings"],
        "rounds": rounds,
    }
