# NeuroCore 0.1.0：给原工具开发 AI 的迁移交接

更新：2026-11-14（香港时间）。仓库：[ecingermany/NeuroCore](https://github.com/ecingermany/NeuroCore)。

本文是接入工作的起点：告诉原报告、分析及训练工具的维护者，应替换哪些算法入口、保留哪些业务代码，以及“结果一致”的验证范围。本文不是让读取文档的 AI 无条件删除原代码或跳过用户的变更要求；执行范围以用户当前授权为准。

## 1. 先理解“已经验证一致”具体指什么

**数值算法已做回归，但尚未把所有原工具切换到 NeuroCore 并完成完整用户流程验收。** 不应把本文解读为所有报告、GUI、软件版本和硬件配置已经全面等价。

| 范围 | 基准与结果 | 尚未覆盖 |
| --- | --- | --- |
| YYG 扩展特征 | 指定原算法的四份120秒合成记录，年龄6/12/30/65；全局及逐秒频谱、显示增益与8类特征结果逐项通过 | 任意历史分支、全部真实人群记录、迁移后的所有工具端到端 |
| 独立 MonitorR 常用模式 | 原JS引擎183块合成基准，模式1→4→2，继承历史；指标、阈值、计数、条件、门控及奖励通过 | 所有模式独立实机会话、所有设备配置及完整低频显示等价 |
| BQRequest 报告指数 | 20组既有真实BFN/BAN，主要指数最大误差约5×10^-5，Synchrony约5×10^-7，与文本保存精度相符 | 其他官方版本、所有异常及年龄边界的实测覆盖 |
| BQ 报告补充项 | 同20组LogBand、Slope、Absolute/Relative Strength通过，活性/情绪标签各20/20 | 所有官方图像像素和未出现的显示标签 |
| BQtester 三项动态测评 | 已知版本一次真实会话：三项P/W与BFN相符，180个奖励与原验证脚本相符 | 旧20组训练P/W跨版本重算、全部屏幕格子与存档差异根因 |
| 三维图与包安装 | 保留YYG原图构造函数；Plotly图、离线HTML、独立环境wheel安装及示例运行成功 | BrainSee/MeasureR像素级复刻及原工具排版全流程 |

本机34项公开测试已通过。完整证据见 [验证记录](validation-2026-11-14.zh-CN.md)、[匿名真实资料回归摘要](private-validation-summary.json)、[源文件哈希](provenance.json) 和 [JS参考来源](monitor-reference.json)。私有原始数据不在仓库内，公开环境不能重跑不存在的个人材料。

四份YYG金样由原实现生成，再与包比较，不是由待测包给自身生成期望值。容差为相对1e-9、绝对1e-10。EQ/PAF业务文案采用原实现的“无建议库回退”基准；保留原工具文案时，必须按第5节注入原建议库。

## 2. 安装与固定版本

发行名是 **neurocore-eeg**，Python导入名是 **neurocore**。本版未发布PyPI；不要直接假设 `pip install neurocore-eeg==0.1.0` 已能从PyPI取得本仓库代码。

推荐在原工具自己的虚拟环境安装仓库固定提交或本地wheel。首次接入可这样做：

```powershell
git clone https://github.com/ecingermany/NeuroCore.git
cd NeuroCore
git rev-parse HEAD
# 将输出的完整commit SHA记录到原工具依赖锁定记录；部署时checkout该SHA
python -m pip install ".[viz]" -c constraints-reference.txt
```

运行测试、重建安装包时先执行：

```powershell
python -m pip install ".[dev,viz]" -c constraints-reference.txt
python -m pytest -q
python -m build
```

生成的wheel为 `dist/neurocore_eeg-0.1.0-py3-none-any.whl`。普通安装不需要开发依赖。部署时固定完整Git提交比长期跟随main更可复现；包内0.1.0版本号不能替代提交锁定。

参考环境：Python3.9.5、NumPy2.0.2、SciPy1.13.1、Plotly7.1.0。Python语法兼容3.9，其他运行版本需在目标环境回归。没有Plotly时核心算法仍可导入；绘图需 `[viz]`。不要为安装包而升级原工具全部依赖。

## 3. 选择正确算法线路

| 原工具的实际需求 | 应用的入口 |
| --- | --- |
| 原YYG单次／跨时间／前后测／群体分析 | `neurocore.analysis.yyg` |
| 官方BQtester三项动态测评 | `neurocore.assessment.bqtester` |
| 官方静态O1/C/O2与训练P/W生成报告指数 | `neurocore.assessment.bqrequest` |
| BrainSee默认报告频谱口径 | `neurocore.assessment.brainsee` |
| MeasureR默认测量频谱和RMS | `neurocore.neurofeedback.measurer` |
| 独立MonitorR日常神经反馈 | `neurocore.neurofeedback.monitorr` |
| 已恢复的游戏计分／奖励消费 | `neurocore.neurofeedback.games` |

**迁移已有YYG工具时，应先替换为YYG兼容接口，不能直接改用官方BQ指数。** 这两类算法的名称相近但定义、阶段和输出并不相同。YYG静态协议120秒，官方BQ静态O1/C/O2共90秒；禁止自动均分或裁剪来伪装成同一协议。

源码位置与导入路径对应，例如 `src/neurocore/assessment/bqrequest.py` 提供 `neurocore.assessment.bqrequest`。不要复制包内源码回工具，也不要在正式业务中依赖内部 `_features.py`、`_feedback.py` 或 `_spectral.py`。

## 4. YYG：旧代码到新API的逐项替换表

YYG抽取基线为 `da014597054a01f95dc820210a663d85737aeb65`。本次检查时main已前进，但提取的feature_analysis.py、dynamic_txt.py和eeg_report文件与该基线无差异，具体哈希见provenance.json。旧9月桌面交付版不能因此认定与当前基线相同。

| 原文件／符号 | 替换后的公开入口 |
| --- | --- |
| feature_analysis.ChannelSpectrum / FeatureResult | `neurocore.analysis.yyg.ChannelSpectrum` / `FeatureResult` |
| load_and_process / load_and_process_dual | `neurocore.analysis.yyg` 同名函数 |
| get_scaling_array / THETA_DROP_ABS_THRESHOLD | `neurocore.analysis.yyg` 同名符号 |
| compute_physiological_tension / compute_theta_emotion / detect_theta_drop_events | `neurocore.analysis.yyg` 同名函数 |
| compute_eq_feature / compute_paf_feature | `neurocore.analysis.yyg` 同名函数 |
| compute_theta_beta_ratio / get_tbr_threshold | `neurocore.analysis.yyg` 同名函数 |
| compute_delta_eyes_change / compute_theta_smr_ratio / compute_delta_smr_ratio | `neurocore.analysis.yyg` 同名函数 |
| get_age_adjusted_tension_threshold / get_age_adjusted_emotion_threshold | `neurocore.analysis.yyg` 同名函数 |
| parse_birthdate / parse_scan_date / compute_age_from_birthdate | `neurocore.analysis.yyg` 同名函数 |
| _compute_dsr_patterns / DSR_PATTERN_ORDER | `neurocore.analysis.yyg.compute_dsr_patterns` / `DSR_PATTERN_ORDER` |
| _tension_closed_rise_detail / _tension_closed_rise_level | `neurocore.analysis.yyg.tension_closed_rise_detail` / `tension_closed_rise_level` |
| _compute_eq_patterns / EQ_PATTERN_ORDER | `neurocore.analysis.yyg.compute_eq_patterns` / `EQ_PATTERN_ORDER` |
| _compute_paf_patterns / _compute_paf_patterns_from_channels / PAF_PATTERN_ORDER | `neurocore.analysis.yyg.compute_paf_patterns` / `compute_paf_patterns_from_channels` / `PAF_PATTERN_ORDER` |
| _build_3d_fft_chart / _build_3d_fft_chart_single | `neurocore.analysis.yyg.visualization.build_3d_fft_chart` / `build_3d_fft_chart_single` |
| dynamic_txt.compute_from_txt等 | `neurocore.analysis.yyg.dynamic_txt` 对应原符号 |
| eeg_report.loader / preprocess / spectral / config / eq / paf / validation | `neurocore.analysis.yyg.eeg_report` 下对应模块及原符号 |

大多数参数和返回结构保留；不要同时更改函数调用参数、滤波模式或读取路径。新数据类的Python模块路径不同，原pickle/跨进程对象的兼容性需单独验收。

### 4.1 跨时间CLI及GUI

eeg_cross_time.py原来从feature_analysis导入五个函数，可改为：

```python
from neurocore.analysis.yyg import (
    load_and_process,
    compute_physiological_tension,
    compute_theta_emotion,
    compute_eq_feature,
    compute_paf_feature,
)
from neurocore.analysis.yyg.eeg_report.validation import (
    validate_file,
    # 该工具正在使用的其他原validation符号仍按原名显式导入
)
```

保持原文件查找、日期计算、报告模板、比较逻辑、绘图布局、GUI和输出命名。运行原跨时间冒烟并比较迁移前后的各次测量数值。

### 4.2 前后测CLI及GUI

eeg_pre_post.py除上述五个函数，还引用THETA_DROP_ABS_THRESHOLD及单通道私有绘图函数：

```python
from neurocore.analysis.yyg import THETA_DROP_ABS_THRESHOLD
from neurocore.analysis.yyg.visualization import (
    build_3d_fft_chart_single as _build_3d_fft_chart_single,
)
```

这个局部别名可暂时保留现有调用点；它指向新包的公开API，不要求全文件一次改名。`traditional`、`config`、`height`等原绘图参数保持原调用值。PDF/打印/HTML排版继续由工具负责。

### 4.3 DSR群体报告和原测试

```python
from neurocore.analysis.yyg import (
    compute_dsr_patterns as _compute_dsr_patterns,
    DSR_PATTERN_ORDER,
)
```

保持 `for_advice` 原值。包还提供部分私有名称的弃用别名作过渡，但新增代码使用公开名称。

### 4.4 单次报告和云端引擎

单次报告的feature_analysis.py仍可保留为工具入口和报告生成层，只从包导入算法。**不要直接删除整个feature_analysis.py**：它还包含CLI、HTML、建议文案及其他工具职责。先把原本地同名算法定义移除或改为薄封装，避免它们在文件后部覆盖新import。

云端先逐项核对engine_files副本与指定真源差异，再替换实际算法import；保留平台鉴权、存储、任务调度、报告模板和建议数据。不根据本库回归通过就删除整个engine_files目录，直到确认其中没有需要保留的业务代码。

## 5. 保持YYG结果一致时最容易漏掉的细节

1. `load_and_process` 原默认仍是 `per_second`。原工具显式传 `global` 时必须保留；不能为了“统一”改默认。
2. `load_and_process_dual` 返回 `(a_global,b_global,a_persecond,b_persecond,seconds)`。按原调用用途选择；PAF需要原Tukey链时保留filepath和原filter_mode，不能只把Hann谱当作PAF输入。
3. 保留年龄、阶段、阈值、缩放和舍入顺序。显示增益是YYG行为，不加到官方报告或训练链。
4. 群体全程紧张与单次分段紧张保持原统计用途，不混用。
5. 原建议库不随包分发。若原报告依赖EQ/PAF补充文案，将原库对象显式注入，否则包使用无库回退文案，数值相同但文字可能不同：

```python
from neurocore.analysis import yyg

# library由原工具按既有规则加载，此处不改变远端更新或年龄分组策略
with yyg.advice_library(library):
    eq_result = yyg.compute_eq_feature(a, b, age=age)
    paf_result = yyg.compute_paf_feature(
        paf_a, paf_b, age=age, filepath=filepath,
        filter_mode=original_filter_mode,
    )
```

6. 既有三维HTML优先用YYG原绘图兼容入口；通用plotting渲染器不保证原图相机、色彩、排版相同。
7. `analysis.yyg.dynamic_txt.compute_from_txt`保留原估算行为：183块按三轮切分、每轮冷启动，缺少完整门控时假定通过并标记estimated。**这是旧工具兼容接口，不是完整门控核验后的官方结果。** 不把它改写为“真实奖励已确认”。
8. 历史解释和mV等标签为兼容而保留，不意味着单位或解释得到新的验证。

## 6. 官方测评报告：替换研究脚本中的计算部分

原 `analysis/bq/validate_bq_raw.py` / `validate_report_extras.py` 是带本地文件和DLL读取的研究验证脚本。生产工具可以用本包纯函数替代其中的数学计算，不整体import这些带顶层I/O的脚本。

```python
from neurocore.assessment.bqtester import evaluate_assessment
from neurocore.assessment.bqrequest import compute_report

# 原始整数ADC，数组顺序必须显式明确
# static: [3,30,2,256]，O1/C/O2
# training: [3,60,2,256]，放松/注意/集中
dynamic = evaluate_assessment(training)
report = compute_report(
    static, age=age,
    percentages=[stage['percentage'] for stage in dynamic['stages']],
    weights=[stage['weight'] for stage in dynamic['stages']],
)
```

已有BFN保存P/W时，可直接传其P/W给compute_report，不要求重新计算。解码文本适配入口：

```python
from neurocore.data.decoded_bq import parse_decoded_bfn

decoded = parse_decoded_bfn(decoded_text)
report = compute_report(decoded.stages[:3], decoded.age,
                        decoded.percentages, decoded.weights)
```

本包不负责厂商加密BFN/BAN的生成、解密、签名、上传或服务器授权。已有解密/格式工作流保留在宿主或单独适配层。

### 输出合同

- 报告主要键：BRQ、SRQ、Rate、AoverBl、ATQ、ACQ、EQ、AntiStress、Synchrony、Symmetry、CQ、BQ。
- 双侧数值按设备A/B顺序；A/B到物理左右位置由宿主配置明确，不偷偷交换。
- 中间量：spectra `[3,30,2,128]`、bands `[3,30,2,6]`、mean_bands `[3,2,6]`、EQ_curve、peak_frequencies。
- 补充量：AbsoluteStrength、RelativeStrength、LogDifference、ArithmeticDifference、activity_label、emotion_label、LogBand `[3,2,6]`、Slope `[3,3,2]`（三类斜率、三阶段、双通道）及训练汇总字段。
- 返回值含NumPy数组，不承诺直接json.dumps；跨服务传输时显式转为list/float。指标先比较未舍入数值，再比较页面显示。
- 当前数据校验拒绝不支持的年龄、无效阶段、非整数/非有限采样和未定义比值。这是本包输入策略，不宣称原软件对坏输入也如此。

## 7. 独立MonitorR：替换JS或旧Python引擎时的合同

旧对应实现：网页dynamic-core.js、dynamic-input.js；Python动态TXT兼容代码另见第5节。以下接口用于完整原始数据训练：

```python
from neurocore.neurofeedback.monitorr import MonitorEngine

engine = MonitorEngine()
# 每通道256个原始设备整数；默认已验证source_type=0x82000000
result = engine.process_raw(raw_a, raw_b, mode=4)
if result['reward'] is True:
    trigger_success_feedback()
saved = engine.snapshot()
engine.reset()       # 清阈值、历史槽、计数和帧数
engine.restore(saved)
```

- 模式1/4/2/5对应放松/SMR/Beta/组合分支；同一实例历史持续，不按模式名自动重置。
- 已有128点处理数据可用 `engine.process(a,b,mode,gate=None)`。候选通过但gate未知时reward为None，不当作成功；原始门控要两侧完整256槽混合缓冲，不能补零。
- `process_raw`支持已验证的0x81/0x82输入适配，保留后128槽；不要把BQtester原始N256或YYG滤波谱直接传入128点接口。
- 原JS的快照没有本包profile字段，不可直接restore；默认保留旧实现直到明确映射并验证迁移时的历史，或在新会话明确冷启动。不要默默丢掉正在训练的状态。
- 三轮历史TXT重建应根据其原始上下文确定继承／重置、61块边界和60格计分；不能套用另一工具默认设置。
- 模式6/7/8用单独 `MeditationEngine`，含特殊共享历史，属于静态恢复。其与常用引擎快照不能互换。

JS页面无法直接安装Python wheel。维持现有JS实时链，或另接Python后端并验证延迟与生命周期；普通import替换只适用于Python项目。不要将后端引入当成此次数值回归已覆盖的网页架构改造。

## 8. 三维图及游戏接入

| 原用途 | 新入口 | 兼容承诺 |
| --- | --- | --- |
| 原YYG单／双通道三维HTML | analysis.yyg.visualization的两个build函数 | 原构造函数正文保留；宿主仍需验收排版 |
| MeasureR默认频谱与RMS | neurofeedback.measurer.spectrum / band_rms / spectrum_grid | 已知N256 Hann配置，不附加YYG增益 |
| BrainSee默认绘图谱 | assessment.brainsee.spectrum / spectrum_grid | 无窗静态证据，不是官方像素级图像 |
| 新工具通用三维图 | plotting.surface_figure / write_html | 读取SpectrumGrid元数据；可选Plotly，离线HTML |

BrainSee、MeasureR的spectrum_grid接收单通道 `[seconds,256]`。不得仅靠给同一张图换标题就宣称算法相同。BQtester实时三维图路径尚未确认，不擅自指定为无窗。

games模块提供的纯规则及对应对象：

- ArrowR：ArrowState、arrow_rms、arrow_reference；结算使用最近E/W和实际配置参照，不把reward直接当环数。
- CupR：CupState；需要真实媒体结束/方向/保留状态，不是每成功固定加8。
- NeuRaceR：race_action；播放／暂停。
- PuzzleR：puzzle_tick；由动画tick调用，不按EEG秒数直接加分。
- SSoundR：memory_result；题目正确性与脑波准入分开。
- FrcDownR：frc_down_feedback；媒体结束分支与普通成功分开。
- MeditR：meditation_action；还考虑提示阻塞。
- NeuroSpectrum等声音消费：reward_sound；每个新的成功结果可触发，不仅奖励上升沿。

这些不含原厂媒体、音频、窗口、输入或完整游戏题目状态机；继续由宿主提供。未分析的游戏和未知配置不补写公式。

## 9. 原工具应保留的内容

保留CLI/GUI、报告HTML模板、PDF/打印流程、品牌／水印／文案／建议库、设备实际串口打开和初始化、档案存储、云端接口与权限、网页采集生命周期及游戏资源。

`data.protocol`只提供已验证N256帧的CRC/解交错纯函数，不自动连接设备或推广初始化profile。`parse_decoded_bfn`仅解析已解码研究文本，不是完整生产格式安全边界。

## 10. 接入顺序与完成标准

1. 记录原工具commit、依赖、输入、年龄、阶段、滤波口径、模式和历史；保留迁移前输出。
2. 在适合修改的原工具副本/分支安装固定NeuroCore提交。先跑本仓库34项测试；CI结果另查，不假设本机通过等于远端已通过。
3. 只改本次需要的算法import和必要适配；用第4节局部别名减少改动。保持报告和交互代码。
4. 在同一输入上比较：频谱矩阵→中间特征→最终结果→页面舍入。官方BAN按其文本精度；YYG包内回归按金样容差。训练P/W为整数，要求逐项精确一致，不能只看SRQ恰好一样。
5. 保留建议库注入和原绘图入口，再验收单次、前后测、跨时间、群体报告及所用GUI/导出。用户已交付的历史报告不回写。
6. 通过后才移除实际已替代的算法副本，固定依赖提交，并记录仍留在工具层的职责；保留可回退的旧版本。

接手AI的交付应包含：更改文件列表、具体import映射、固定提交/依赖、同输入差异报告、原工具冒烟结果、剩余范围。出现差异先确认口径和状态，不调整公式去拟合预期数值，不把某次样本匹配推广成所有版本等价。

## 11. 可直接交给其他AI的简短任务说明

> 请先读NeuroCore仓库的docs/AI-MIGRATION-HANDOFF.zh-CN.md、docs/integration.zh-CN.md及验证记录。将当前工具实际使用的算法迁移到对应公开API，优先保留参数、返回结构、滤波口径、历史状态、建议库和绘图行为。不要整体删除feature_analysis.py或平台业务目录，不要在YYG、BQtester、BQRequest和独立MonitorR之间混用算法。先进行固定输入的迁移前后数值对照及原工具冒烟，确认通过后再清理被替代副本。NeuroCore已有数值验证不等于当前工具已完成接入验收。
