"""Global configuration for EEG signal processing."""


SAMPLE_RATE = 256
FRAME_SIZE = 256
FREQ_RESOLUTION = SAMPLE_RATE / FRAME_SIZE  # 1 Hz

NOTCH_BAND = [47.5, 52.5]
NOTCH_ORDER = 4

BANDPASS_BAND = [0.5, 40.0]
BANDPASS_ORDER = 4

ALPHA_BAND = [8, 12]
PAF_FREQ_RANGE = [3, 14]
PAF_BRQ_FREQ_RANGE = [3, 15]
PAF_FULL_FREQ_MAX = 29  # 0-29Hz for peak counting

# ── 旧版固定绝对阈值（MATLAB 遗产，保留作对照/回退，不再默认使用）──
# 2026-10-09 在 7 份标注数据上的统计发现其失效：这批数据 3-15Hz 幅度中位数
# 6.3-131.6，89-95% 频点超过 2.0，每帧平均检出 2.5-3.5 个"峰"，直方图被
# 噪声票垫平；无峰数据（ccym-l14）的 PAF 沦为噪声抽样但仍输出看似正常的值。
# 统计细节见 docs/20261009_paf_prominence_dynamic_threshold_log.md。
PAF_PEAK_THRESHOLD = 2.0

# ── PAF 无阈值投票（2026-10-09 下午定稿）──
# 投票规则：PAF_VOTE_BAND 带内局部极大直接计票，不设任何幅度阈值。
# 演进史：固定绝对阈值 2.0（MATLAB 遗产，失效统计见上）→ C2 动态阈值（同日上午）
# → 同日下午彻底删除阈值。删除依据（6 份数据 12 通道统计，output/_paf_diag_no_threshold.py）：
#   1. 3-15Hz 带内每帧局部极大仅 3-5 个，带内频谱平滑、起伏天然稀少；
#   2. 固定 2.0 带内过滤率为 0（123-143 个局部极大近乎 100% 通过）——阈值在带内空转，
#      历史作用只是挡 0-29Hz 全频计数时的高频噪声（15-29Hz 幅度仅为带内 0.10-0.18 倍）；
#   3. 幅度阈值（固定或自适应）对"平台型"alpha 有系统性盲区：gaoya 前额 11-12Hz 相对
#      1/f 趋势抬升 21-48%（真实但宽而弱），被 C2 的中位数底线抹平；
#   4. 无阈值下 CKK 4 通道仍独领先 10Hz（20-24 票 vs 次高 12-14），gaoya 回到 11/11Hz
#      （与受试者外部先验一致），异常帧（缩放>1.5）带内不产生额外结构（3-4 峰/帧，与正常帧同）。
# 真峰靠直方图时间维度累积浮现；噪声票随机散布相互抵消。可靠性由峰突出度闸门独立把关。
PAF_VOTE_BAND = [3, 15]

# ── PAF 峰突出度闸门（平均谱主峰形态指标，独立于直方图投票路径）──
# 突出度 = (峰高 - 鞍部) / 鞍部 × 100%。鞍部为相邻峰间谷（2026-10-09 下午修订：
# 取代 scipy 延伸鞍部——后者会穿过次峰延伸到远处深谷，双峰时次峰被吞、
# 浅谷双尖被虚高；主峰取 5-12Hz 内峰高最高者，峰数不设上限，多峰形态由
# 突出度数学自然判定）。平均本身就是时间稳定性过滤器：只在少数帧出现的峰
# 会被平均稀释，只有反复出现的真峰才能幸存。与 PAF 选择算法完全独立
# （不用阈值投票、不用年龄先验），作可靠性裁判。
# 产品化为二元判定（2026-10-09 定稿）：突出度数值不进报告，只分两种情况——
#   ≥30% 该侧"检出明显稳定主峰"→ 丘脑处理信息功能稳定；
#   <30% 该侧"未检出明显稳定主峰"→ 对应信息处理可能出现波动，且该侧 PAF
#        不参与左右脑快慢比较与觉醒度评价。
# 定档锚点（7 份数据实测）：CKK 天花板 38-87% 判"有"；ccym 7-14% 或无峰、
# gaoya/Eric 2-8% 判"无"。30% 落在实测空档正中（最高"无"14% vs 最低"有"38%）。
PAF_PROM_BAND = [3, 15]        # 谷搜索频段（相邻谷可延伸至带边，即 5-12Hz 以外数据）
PAF_PROM_PEAK_RANGE = [5, 12]  # 主峰搜索范围（PAF 生理出现范围 5-12Hz）
PAF_PROM_SIGNIFICANT = 30.0    # ≥30% 判"有明显稳定主峰"（二元判定唯一阈值）

PAF_WINDOW_TYPE = "tukey"

# EQ reduction factor: amplitudes are divided by this before L-R subtraction.
# Set to 1.0 for no reduction. Suspected reference value: 2.0
EQ_REDUCTION_FACTOR = 2.0

CLOSED_EYE_START = 40  # 0-indexed second
CLOSED_EYE_END = 80    # 0-indexed second (exclusive)

TUKEY_ALPHA = 0.1

MAX_DISPLAY_FREQ = 30  # Hz, for spectrogram visualization
