"""EEG TXT 檔案完整性驗證模組（跨工具共享）。

從 eeg_cross_time.py 抽出的檔案驗證基礎設施，供跨時間分析、
前後測分析等所有需要讀取 EEG TXT 檔的工具共用，
保證各工具對「有效檔案」的判定完全一致。

硬體協議標準：256 Hz、120 秒、雙通道、總樣本 30720、10-bit ADC 值域 [0, 1023]。

驗證清單（必檢項，任一失敗 → INVALID）:
    1.  檔案頭格式: 含 "Raw Data File" 標記
    2.  日期完整: Start/End Time 均能解析且格式合法
    3.  時間順序: Start < End
    4.  錄製時長: 嚴格等於 120 秒
    5.  通道數 = 2
    6.  Sample Per Line = 256
    7.  Total Sample = 30720
    8.  幀數: A/B 各 120 幀
    9.  偏移序列: 0, 256, 512, ..., 30464 連續無跳幀/重複/亂序
    10. 末幀偏移: A/B 均為 30464，且後跟 256 個整數資料點
    11. 資料行完整: 每個資料區塊行恰好 256 個整數值
    12. 數值範圍: 全部在 [0, 1023]（10-bit ADC）

補充檢查（警告項，不影響有效性判定）:
    - 區塊順序: A/B 嚴格交替且 A 在前
    - 檔名日期 vs 檔案頭日期一致性
    - 平直幀: 整幀所有值相同（電極脫落特徵）
    - 課程編號缺號: 檔名含 L{N} 編號時偵測缺失序號

另外包含同時間點重複偵測（DUPLICATE 狀態，僅保留一份）。
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional


# ── 标准参数（硬件协议） ──────────────────────────────────────

STANDARD_SAMPLE_RATE = 256
STANDARD_PROTOCOL_SECONDS = 120
STANDARD_TOTAL_SAMPLES = STANDARD_SAMPLE_RATE * STANDARD_PROTOCOL_SECONDS  # 30720
LAST_FRAME_OFFSET = STANDARD_TOTAL_SAMPLES - STANDARD_SAMPLE_RATE          # 30464
ADC_MIN, ADC_MAX = 0, 1023        # 10-bit ADC 值域

EXPECTED_OFFSETS = list(range(0, STANDARD_TOTAL_SAMPLES, STANDARD_SAMPLE_RATE))


# ── 数据结构 ──────────────────────────────────────────────────

class FileStatus(Enum):
    COMPLETE = "complete"   # 所有必检项通过
    INVALID = "invalid"     # 至少一项必检失败
    DUPLICATE = "duplicate" # 同一时间点已有记录


class CheckLevel(Enum):
    MANDATORY = "必檢"
    WARNING = "警告"


@dataclass
class Check:
    """单项验证结果。"""
    name: str
    level: CheckLevel
    passed: bool
    detail: str = ""


@dataclass
class EEGFileInfo:
    """单个 EEG 文件的解析与验证结果。"""
    file_path: str
    file_name: str
    file_size: int

    # 文件名解析
    filename_date: Optional[str] = None   # 文件名中的日期 YYYYMMDD
    session_code: str = ""                # 课程编号，如 P2L1AF
    lesson_no: Optional[int] = None       # 课程序号，如 1

    # 文件头
    header_start_time: Optional[datetime] = None
    header_end_time: Optional[datetime] = None
    sample_rate: int = 0
    total_sample_declared: int = 0
    channel_count: int = 0

    # 数据验证
    n_frames_a: int = 0
    n_frames_b: int = 0
    last_offset_a: Optional[int] = None
    last_offset_b: Optional[int] = None
    recording_time: Optional[datetime] = None

    # 验证明细
    checks: List[Check] = field(default_factory=list)
    status: FileStatus = FileStatus.INVALID

    @property
    def is_usable(self) -> bool:
        return self.status == FileStatus.COMPLETE

    @property
    def failed_mandatory(self) -> List[Check]:
        return [c for c in self.checks
                if not c.passed and c.level == CheckLevel.MANDATORY]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks
                if not c.passed and c.level == CheckLevel.WARNING]


@dataclass
class ScanResult:
    """目录扫描汇总结果（一个目录 = 同一受测者的时间序列）。"""
    directory: str
    all_files: List[EEGFileInfo] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.all_files)

    @property
    def valid_sessions(self) -> List[EEGFileInfo]:
        """按时间排序、可用于分析的有效扫描序列（完整且非重复）。"""
        return [f for f in self.all_files if f.status == FileStatus.COMPLETE]

    @property
    def invalid_files(self) -> List[EEGFileInfo]:
        return [f for f in self.all_files if f.status == FileStatus.INVALID]

    @property
    def duplicates(self) -> List[EEGFileInfo]:
        return [f for f in self.all_files if f.status == FileStatus.DUPLICATE]

    @property
    def time_span_days(self) -> Optional[int]:
        sessions = self.valid_sessions
        times = [s.recording_time for s in sessions if s.recording_time]
        if len(times) >= 2:
            return (times[-1] - times[0]).days
        return None

    def missing_lessons(self) -> List[int]:
        """课程编号缺号检测（文件名含 L{N} 时）。"""
        lessons = sorted(s.lesson_no for s in self.valid_sessions
                         if s.lesson_no is not None)
        if len(lessons) < 2:
            return []
        return [n for n in range(lessons[0], lessons[-1] + 1) if n not in lessons]


# ── 文件名解析 ────────────────────────────────────────────────

# 课程编号 P2L17AF / P1L3 等
_RE_SESSION_CODE = re.compile(r"\bP(\d+)L(\d+)([A-Za-z]*)\b")
# YYYYMMDD 开头或结尾
_RE_DATE_PREFIX = re.compile(r"^(\d{8})\b")
_RE_DATE_SUFFIX = re.compile(r"\b(\d{8})$")


def parse_filename(file_name: str) -> dict:
    """从文件名解析日期和课程编号（不做受测者身份识别）。

    支持的命名模式:
        YYYYMMDD-EP-HKKT-P2L1AF.txt → 日期 + 课程编号
        YYYYMMDD-Celine.txt          → 日期
        Celine-20260904.txt          → 日期
        [ADHD]ID_WongTszYin.txt      → 无日期（时间从文件头获取）

    Returns:
        {"date": str|None, "session_code": str, "lesson_no": int|None}
    """
    stem = Path(file_name).stem
    out = {"date": None, "session_code": "", "lesson_no": None}

    m = _RE_DATE_PREFIX.match(stem)
    if m:
        out["date"] = m.group(1)
    else:
        m = _RE_DATE_SUFFIX.search(stem)
        if m:
            out["date"] = m.group(1)

    sm = _RE_SESSION_CODE.search(stem)
    if sm:
        out["session_code"] = sm.group(0).upper()
        out["lesson_no"] = int(sm.group(2))

    return out


# ── 文件头解析 ────────────────────────────────────────────────

_RE_TIME_LINE = re.compile(
    r"(\w+)\.\s*(\w+)\s+(\d{1,2})\.(\d{4}),\s*(\d{2}):(\d{2}):(\d{2})\s+(Start|End)\s+Time",
    re.IGNORECASE
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_time_line(line: str) -> Optional[datetime]:
    """解析 "Thursday. May 28.2026,16:28:41 Start Time" 格式的时间行。"""
    m = _RE_TIME_LINE.search(line)
    if not m:
        return None
    month = _MONTH_MAP.get(m.group(2).lower())
    if not month:
        return None
    try:
        return datetime(
            year=int(m.group(4)), month=month, day=int(m.group(3)),
            hour=int(m.group(5)), minute=int(m.group(6)), second=int(m.group(7)),
        )
    except (ValueError, IndexError):
        return None


def parse_header(lines: List[str]) -> dict:
    """解析 TXT 文件头部元数据。"""
    meta = {
        "is_valid_format": False,
        "start_time": None,
        "end_time": None,
        "channel_count": 0,
        "sample_rate": 0,
        "total_sample": 0,
        "data_start_line": -1,
    }

    for i, line in enumerate(lines):
        s = line.strip().lstrip("\ufeff")

        if s == "Raw Data File":
            meta["is_valid_format"] = True
            continue
        if "Start Time" in s:
            meta["start_time"] = _parse_time_line(s)
            continue
        if "End Time" in s:
            meta["end_time"] = _parse_time_line(s)
            continue
        if s.startswith("Channel Number:"):
            try:
                meta["channel_count"] = int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
            continue
        if s.startswith("Sample Per Line:"):
            try:
                meta["sample_rate"] = int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
            continue
        if s.startswith("Total Sample:"):
            try:
                meta["total_sample"] = int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
            continue
        if re.match(r"^A\s+\d+$", s):
            meta["data_start_line"] = i
            break

    return meta


# ── 数据区解析 ────────────────────────────────────────────────

_RE_BLOCK_MARKER = re.compile(r"^([AB])\s+(\d+)$")


def parse_data_region(lines: List[str], start_idx: int) -> dict:
    """遍历数据区，统计偏移序列、数据行完整性与平直帧。

    Returns:
        {"offsets_a", "offsets_b", "marker_seq",
         "bad_len_lines", "non_int_lines", "out_of_range",
         "min_val", "max_val", "flat_frames"}
    """
    stats = {
        "offsets_a": [], "offsets_b": [], "marker_seq": [],
        "bad_len_lines": 0, "non_int_lines": 0, "out_of_range": 0,
        "min_val": None, "max_val": None, "flat_frames": 0,
    }

    i = start_idx
    n = len(lines)

    while i < n:
        m = _RE_BLOCK_MARKER.match(lines[i].strip())
        if not m:
            i += 1
            continue

        ch, offset = m.group(1), int(m.group(2))
        stats["marker_seq"].append(ch)
        stats["offsets_a" if ch == "A" else "offsets_b"].append(offset)

        if i + 1 < n:
            parts = lines[i + 1].strip().split(",")
            if len(parts) != STANDARD_SAMPLE_RATE:
                stats["bad_len_lines"] += 1
            else:
                try:
                    vals = [int(v) for v in parts]
                except ValueError:
                    stats["non_int_lines"] += 1
                else:
                    vmin, vmax = min(vals), max(vals)
                    stats["min_val"] = vmin if stats["min_val"] is None else min(stats["min_val"], vmin)
                    stats["max_val"] = vmax if stats["max_val"] is None else max(stats["max_val"], vmax)
                    stats["out_of_range"] += sum(1 for v in vals
                                                 if v < ADC_MIN or v > ADC_MAX)
                    if vmin == vmax:
                        stats["flat_frames"] += 1
        else:
            stats["bad_len_lines"] += 1

        i += 2

    return stats


# ── 单文件验证 ────────────────────────────────────────────────

def validate_file(file_path: str) -> EEGFileInfo:
    """对单个 TXT 文件执行完整验证，返回带检查清单的 FileInfo。"""
    p = Path(file_path)
    info = EEGFileInfo(
        file_path=str(p),
        file_name=p.name,
        file_size=p.stat().st_size if p.exists() else 0,
    )
    checks = info.checks

    def add(name, level, passed, detail=""):
        checks.append(Check(name, level, passed, detail))

    if not p.exists():
        add("檔案存在", CheckLevel.MANDATORY, False, "檔案不存在")
        return info

    # 文件名解析
    fn = parse_filename(p.name)
    info.filename_date = fn["date"]
    info.session_code = fn["session_code"]
    info.lesson_no = fn["lesson_no"]

    # 读取文件
    try:
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError as e:
        add("檔案可讀取", CheckLevel.MANDATORY, False, str(e))
        return info

    # ── 檔案頭 ──
    header = parse_header(lines)

    add("檔案頭格式", CheckLevel.MANDATORY, header["is_valid_format"],
        "" if header["is_valid_format"] else "缺少 Raw Data File 標記")
    if not header["is_valid_format"]:
        return info

    info.header_start_time = header["start_time"]
    info.header_end_time = header["end_time"]
    info.channel_count = header["channel_count"]
    info.sample_rate = header["sample_rate"]
    info.total_sample_declared = header["total_sample"]

    dates_ok = header["start_time"] is not None and header["end_time"] is not None
    add("日期完整", CheckLevel.MANDATORY, dates_ok,
        "" if dates_ok else
        f"Start={header['start_time']}, End={header['end_time']}（存在無法解析的時間）")

    order_ok = dates_ok and header["start_time"] < header["end_time"]
    add("時間順序", CheckLevel.MANDATORY, order_ok,
        "" if order_ok else "Start Time 不早於 End Time")

    if dates_ok:
        duration = (header["end_time"] - header["start_time"]).total_seconds()
        add("錄製時長=120s", CheckLevel.MANDATORY, duration == 120.0,
            f"實際 {duration:g}s" if duration != 120.0 else "")

    add("通道數=2", CheckLevel.MANDATORY, header["channel_count"] == 2,
        f"實際 {header['channel_count']}" if header["channel_count"] != 2 else "")
    add("取樣率=256", CheckLevel.MANDATORY, header["sample_rate"] == STANDARD_SAMPLE_RATE,
        f"實際 {header['sample_rate']}" if header["sample_rate"] != STANDARD_SAMPLE_RATE else "")
    add("總樣本=30720", CheckLevel.MANDATORY, header["total_sample"] == STANDARD_TOTAL_SAMPLES,
        f"實際 {header['total_sample']}" if header["total_sample"] != STANDARD_TOTAL_SAMPLES else "")

    if header["data_start_line"] < 0:
        add("資料區存在", CheckLevel.MANDATORY, False, "未找到資料起始標記（A  0）")
        return info

    # ── 資料區 ──
    stats = parse_data_region(lines, header["data_start_line"])

    info.n_frames_a = len(stats["offsets_a"])
    info.n_frames_b = len(stats["offsets_b"])
    info.last_offset_a = stats["offsets_a"][-1] if stats["offsets_a"] else None
    info.last_offset_b = stats["offsets_b"][-1] if stats["offsets_b"] else None

    frames_ok = (info.n_frames_a == STANDARD_PROTOCOL_SECONDS
                 and info.n_frames_b == STANDARD_PROTOCOL_SECONDS)
    add("幀數=120", CheckLevel.MANDATORY, frames_ok,
        f"A={info.n_frames_a}, B={info.n_frames_b}" if not frames_ok else "")

    seq_ok = (stats["offsets_a"] == EXPECTED_OFFSETS
              and stats["offsets_b"] == EXPECTED_OFFSETS)
    seq_detail = ""
    if not seq_ok:
        parts = []
        for label, offs in [("A", stats["offsets_a"]), ("B", stats["offsets_b"])]:
            if offs != EXPECTED_OFFSETS:
                bad = [i for i in range(min(len(offs), len(EXPECTED_OFFSETS)))
                       if offs[i] != EXPECTED_OFFSETS[i]]
                pos = f"位置 {bad[0]}" if bad else "長度"
                parts.append(f"{label} 通道偏移異常（首個異常{pos}）")
        seq_detail = "; ".join(parts)
    add("偏移序列連續", CheckLevel.MANDATORY, seq_ok, seq_detail)

    last_ok = (info.last_offset_a == LAST_FRAME_OFFSET
               and info.last_offset_b == LAST_FRAME_OFFSET)
    add("末幀偏移=30464", CheckLevel.MANDATORY, last_ok,
        f"A 末幀 {info.last_offset_a}, B 末幀 {info.last_offset_b}" if not last_ok else "")

    lines_ok = stats["bad_len_lines"] == 0 and stats["non_int_lines"] == 0
    add("資料行完整", CheckLevel.MANDATORY, lines_ok,
        f"長度異常 {stats['bad_len_lines']} 行, 非整數 {stats['non_int_lines']} 行"
        if not lines_ok else "")

    range_ok = (stats["min_val"] is not None
                and stats["out_of_range"] == 0)
    add("數值範圍[0,1023]", CheckLevel.MANDATORY, range_ok,
        f"越界值 {stats['out_of_range']} 個" if not range_ok else "")

    # ── 警告級檢查 ──
    expected_seq = "AB" * STANDARD_PROTOCOL_SECONDS
    seq_str = "".join(stats["marker_seq"])
    add("區塊順序交替", CheckLevel.WARNING, seq_str == expected_seq,
        f"實際 {seq_str[:20] + '...' if len(seq_str) > 20 else seq_str}"
        if seq_str != expected_seq else "")

    if info.filename_date and info.header_start_time:
        try:
            name_date = datetime.strptime(info.filename_date, "%Y%m%d").date()
        except ValueError:
            name_date = None
        if name_date:
            header_date = info.header_start_time.date()
            add("檔名日期一致", CheckLevel.WARNING, name_date == header_date,
                f"檔名 {name_date} vs 檔案頭 {header_date}"
                if name_date != header_date else "")

    add("無平直幀", CheckLevel.WARNING, stats["flat_frames"] == 0,
        f"{stats['flat_frames']} 個平直幀" if stats["flat_frames"] else "")

    # ── 录制时间与综合状态 ──
    info.recording_time = info.header_start_time
    if info.recording_time is None and info.filename_date:
        try:
            info.recording_time = datetime.strptime(info.filename_date, "%Y%m%d")
        except ValueError:
            pass

    info.status = (FileStatus.COMPLETE
                   if not info.failed_mandatory else FileStatus.INVALID)

    return info


# ── 目录扫描 ──────────────────────────────────────────────────

def scan_directory(directory: str, recursive: bool = True,
                   progress_cb=None) -> ScanResult:
    """扫描目录中的所有 EEG TXT 文件并验证（一个目录 = 一个人的时间序列）。

    Args:
        directory: 要扫描的目录路径
        recursive: 是否递归扫描子目录
        progress_cb: 可选回呼 progress_cb(當前序號, 總數, 檔名)，供 GUI 進度條使用

    Returns:
        ScanResult（文件按录制时间排序）
    """
    dir_path = Path(directory)
    result = ScanResult(directory=str(dir_path))

    if not dir_path.exists():
        print(f"[錯誤] 目錄不存在: {dir_path}")
        return result

    glob_pattern = "**/*.txt" if recursive else "*.txt"
    txt_files = sorted(dir_path.glob(glob_pattern))

    for i, f in enumerate(txt_files, 1):
        if progress_cb:
            progress_cb(i, len(txt_files), f.name)
        result.all_files.append(validate_file(str(f)))

    result.all_files.sort(key=lambda x: x.recording_time or datetime.min)

    # 标记同时间点重复（精确到秒）
    seen_times = set()
    for s in result.all_files:
        if s.recording_time is None or s.status != FileStatus.COMPLETE:
            continue
        tkey = s.recording_time.strftime("%Y%m%d%H%M%S")
        if tkey in seen_times:
            s.status = FileStatus.DUPLICATE
        else:
            seen_times.add(tkey)

    return result


# ── 报告输出 ──────────────────────────────────────────────────

_STATUS_ICON = {
    FileStatus.COMPLETE: "OK ",
    FileStatus.INVALID: "BAD",
    FileStatus.DUPLICATE: "DUP",
}


def print_scan_summary(result: ScanResult):
    """列印掃描結果摘要。"""
    print("=" * 78)
    print(f"  EEG 檔案掃描與完整性驗證 — {result.directory}")
    print("=" * 78)
    n_ok = len(result.valid_sessions)
    n_bad = len(result.invalid_files)
    n_dup = len(result.duplicates)
    print(f"\n  總檔案數: {result.total_count} | 有效: {n_ok} | "
          f"無效: {n_bad} | 重複: {n_dup}")

    if result.time_span_days is not None:
        print(f"  有效資料時間跨度: {result.time_span_days} 天")

    missing = result.missing_lessons()
    if missing:
        print(f"  ⚠ 課程編號缺號: {', '.join('L' + str(m) for m in missing)}")

    # 無效檔案明細
    if result.invalid_files:
        print(f"\n  ── 無效檔案明細（{len(result.invalid_files)}）──")
        for f in result.invalid_files:
            print(f"\n    {f.file_name}:")
            for c in f.failed_mandatory:
                print(f"      [必檢失敗] {c.name}: {c.detail}")

    # 全部檔案時間線
    print(f"\n  ── 時間序列 ──")
    for s in result.all_files:
        time_str = (s.recording_time.strftime("%Y-%m-%d %H:%M")
                    if s.recording_time else "未知時間")
        sess = f" {s.session_code}" if s.session_code else ""
        print(f"    [{_STATUS_ICON[s.status]}] {time_str}{sess}  {s.file_name}")

    # 警告彙總
    warned = [s for s in result.valid_sessions if s.warnings]
    if warned:
        print(f"\n  ── 警告（{len(warned)} 個有效檔案帶警告）──")
        for s in warned:
            for c in s.warnings:
                print(f"      {s.session_code or s.file_name} — {c.name}: {c.detail}")

    print("\n" + "=" * 78)
