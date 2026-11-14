# NeuroCore

可安装的 Python EEG 核心库：官方测评报告、神经反馈训练、频谱可视化和 YYG 扩展分析。

发行名 **neurocore-eeg**，导入名 **neurocore**，当前版本 **0.1.0**。本地 wheel 交付，尚未发布 PyPI。

**维护已有工具的开发者／AI请先读：[迁移交接文档](docs/AI-MIGRATION-HANDOFF.zh-CN.md)**，包含旧文件／符号到新API的替换表、同输入验证范围、文案与三维图保留方式。

## 安装

需要 Python 3.9 或更高版本。

```powershell
python -m pip install "./dist/neurocore_eeg-0.1.0-py3-none-any.whl[viz]" -c constraints-reference.txt
```

不绘图时去掉 `[viz]`。核心依赖 NumPy、SciPy；可选绘图依赖 Plotly。数值回归环境为 Python 3.9.5、NumPy 2.0.2、SciPy 1.13.1、Plotly 7.1.0；其他版本在 CI 或本机回归前不宣称逐点相同。

## 功能边界

| 模块 | 算法及状态 |
| --- | --- |
| `assessment.bqtester` | 当前已知版本 N256 无窗、每项冷启动，三项训练奖励、P/W、SRQ |
| `assessment.bqrequest` | N256 Hamming，BRQ/ATQ/ACQ/EQ/CQ/SRQ/BQ 等及 LogBand、Slope |
| `assessment.brainsee` | 默认无窗频谱和三维网格；静态代码证据，不承诺官方像素级还原 |
| `neurofeedback.measurer` | N256 periodic Hann、七频带 RMS、三维网格 |
| `neurofeedback.monitorr` | 独立 N128 重建谱、输入适配、动态阈值、状态恢复、完整门控；不混入 BQtester |
| `neurofeedback.games` | 射箭及其他游戏的纯奖励消费／计分规则；画面、媒体和用户输入由应用负责 |
| `analysis.yyg` | 原算法提取，保留参数与返回结构，包括紧张、情绪、EQ、PAF、比值分析和 DSR |
| `plotting` | 可选 Plotly 交互三维图、离线 HTML；YYG 原图兼容接口另保留 |

独立 MonitorR 的模式1/4/2有实测依据，模式5有公共组合证据但未完成独立UI验收。模式6/7/8通过单独的 `MeditationEngine` 提供静态恢复版本，包含共享历史的特殊逻辑，尚未完成完整实机会话验收。

## 测评与报告

```python
from neurocore.assessment.bqtester import evaluate_assessment
from neurocore.assessment.bqrequest import compute_report

# static_stages: [3,30,2,256]，依次首次睁眼/闭眼/再次睁眼
# training_stages: [3,60,2,256]，依次放松/注意/集中，均为原始整数码
training = evaluate_assessment(training_stages)
report = compute_report(
    static_stages, age=30,
    percentages=[s['percentage'] for s in training['stages']],
    weights=[s['weight'] for s in training['stages']],
)
```

## 日常训练

```python
from neurocore.neurofeedback.monitorr import MonitorEngine

engine = MonitorEngine()
result = engine.process_raw(raw_a, raw_b, mode=4)  # 各256点；执行N128适配和完整门控
saved = engine.snapshot()
engine.reset()                                  # 明确重置全部历史
engine.restore(saved)                           # 明确恢复同算法配置的历史
```

输入只有128点TXT时，可调用 `process(a,b,mode,gate=None)`；缺失门控上下文不会被假定为通过。只有 confirmed `reward is True` 才驱动成功反馈。

## 三维图

```python
from neurocore.neurofeedback.measurer import spectrum_grid
from neurocore.plotting import write_html

write_html(spectrum_grid(raw_a_seconds), 'spectrum.html')  # [seconds,256]
```

报告显示口径改用 `assessment.brainsee.spectrum_grid`。图中的“内部幅值”尚未物理电压标定；插值不会增加真实频率分辨率。BrainSee 与 MeasureR 采用不同窗函数，不能靠固定增益互换。

## YYG 原工具接入

```python
from neurocore.analysis.yyg import load_and_process_dual, compute_physiological_tension
from neurocore.analysis.yyg.visualization import build_3d_fft_chart_single

a, b, paf_a, paf_b, seconds = load_and_process_dual('recording.txt')
feature = compute_physiological_tension(a, b, age=30)
html_fragment = build_3d_fft_chart_single(a, seconds, 'recording.txt')
```

完整接入说明见 [docs/integration.zh-CN.md](docs/integration.zh-CN.md)。现有工具的模板、CLI/GUI、建议库和业务文案继续由工具维护；本库不自动修改它们。

## 验证与构建

```powershell
python -m pip install -e ".[dev,viz]" -c constraints-reference.txt
python -m pytest -q
python -m build
```

测试仅使用公开合成夹具；已有本机私有证据的匿名验证摘要另见 [验证说明](docs/validation-2026-11-14.zh-CN.md)。安装包不包含原始个人数据、厂商DLL、WAV或解密密钥，也不生成官方加密BFN/BAN，不依赖厂商安装目录。

源码中保留的历史解释、单位及软件标签用于兼容，不能作为物理标定或临床诊断依据。
