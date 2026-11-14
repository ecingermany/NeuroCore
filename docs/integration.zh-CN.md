# NeuroCore 0.1.0 接入说明

## 1. 本次迁移边界

原项目不修改、不删除算法副本；先在工具自己的虚拟环境安装 wheel，并运行数值和报告冒烟，再逐个修改 import。不要将所有 feature_analysis 的导入整体替换：它原来还包含报告生成和交互函数，这些继续由工具提供。

安装名 neurocore-eeg，导入名 neurocore。推荐固定 `neurocore-eeg==0.1.0` 并记录 constraints-reference.txt 中的数值环境。普通 pip 安装会下载缺少的依赖，完全离线部署需另备对应 Python/平台的 NumPy、SciPy、Plotly wheels；本次 wheel 不嵌入这些二进制依赖。

## 2. YYG 兼容接口

| 原符号 | 新入口 |
| --- | --- |
| load_and_process / load_and_process_dual | neurocore.analysis.yyg 同名 |
| compute_physiological_tension / compute_theta_emotion | neurocore.analysis.yyg 同名 |
| compute_eq_feature / compute_paf_feature | neurocore.analysis.yyg 同名 |
| compute_theta_beta_ratio / compute_delta_eyes_change | neurocore.analysis.yyg 同名 |
| compute_theta_smr_ratio / compute_delta_smr_ratio | neurocore.analysis.yyg 同名 |
| _compute_dsr_patterns | neurocore.analysis.yyg.compute_dsr_patterns |
| _tension_closed_rise_detail / _level | neurocore.analysis.yyg.tension_closed_rise_detail / tension_closed_rise_level |
| _build_3d_fft_chart / _single | neurocore.analysis.yyg.visualization.build_3d_fft_chart / build_3d_fft_chart_single |
| dynamic_txt.compute_from_txt | neurocore.analysis.yyg.dynamic_txt.compute_from_txt |
| eeg_report 底层模块 | neurocore.analysis.yyg.eeg_report 对应模块 |

ChannelSpectrum、FeatureResult 和既有参数/返回结构保留。load_and_process 原有默认仍是 per_second，dual 分别返回全局与逐秒两组，不能在迁移时互换。新类的 Python 模块路径已变化，因此旧 pickle 不保证兼容；跨工具交换优先使用显式数组/JSON。

建议库是业务数据，不进核心。特征函数中原有按库补充的 EQ/PAF 文案通过 context 注入：

```python
from neurocore.analysis import yyg
from advice_loader import AdviceLibrary  # 原工具保留

library = AdviceLibrary()
library.load(update_remote=False)
with yyg.advice_library(library):
    eq = yyg.compute_eq_feature(a, b, age=30)
    paf = yyg.compute_paf_feature(pa, pb, age=30, filepath=path)
```

未注入时使用原函数无库时的回退文案；数值和模式逻辑不受影响。文案注入按ContextVar隔离，避免不同请求互相污染。新的公开名称需要改 import；未全局注入顶层 feature_analysis 或 eeg_report，防止遮蔽原工具报告模块。

## 3. 官方测评

- evaluate_training 仅接收一项 [60,2,256] 原始数据，每次新建冷启动引擎。
- evaluate_assessment 接收三项，顺序1/4/2，内部各自重置。
- compute_report 接收严格 [3,30,2,256] 静态 O1/C/O2 与三个 P/W。
- BQ报告协议90秒与YYG静态120秒不同，不自动按三等分猜测阶段。
- 数据已在BFN中时可用 data.decoded_bq.parse_decoded_bfn 解析已解码文本；解密和签名验证不属于此入口。
- 指标返回未截顶、未按显示精度舍入的内部结果，工具自己负责展示格式。

## 4. 神经反馈与游戏

MonitorEngine 实例保留历史，reset明确清零；process_raw 支持已验证0x81/0x82的256→128适配。process接收已处理128点及已知gate或None。128点TXT没有完整门控，不可补零或默认通过。

mode1/4/2/5是MonitorEngine接口。需要静态恢复的mode6/7/8时显式使用MeditationEngine，其profile与四模式引擎不同，不接受互相导入的快照。mode5、冥想和各游戏完整UI时序不因打包而新增实机验证。

games模块提供ArrowState、arrow_rms、arrow_reference、CupState、race_action、puzzle_tick、memory_result、frc_down_feedback、meditation_action和reward_sound。调用方提供真实媒体/任务状态，计分时机仍由应用决定。它们不代表Windows游戏、动画、音频和全部任务状态机已复刻。没有移交证据的游戏不补写公式。

## 5. 可视化

BrainSee和MeasureR各自生成SpectrumGrid，共用plotting的可选Plotly渲染。这个渲染器是可移植的展示组件，不是官方软件像素复刻，也不宣称复原BQtester尚未确认的实时3D链。

YYG原有3D HTML构造函数单独保留，用于尽量保持既有报告外观。它仍保留旧单位标签和配置，不能据此认定物理标定已完成。

## 6. 发布与验收

先在原工具副本中切换import，检查单次/前后测/跨时间/群体报告数值、图表、交互和导出，再移除冗余副本。本次只完成核心包与接入示例，未声称上述工具已切换或完成端到端验收。

纯浏览器JS不能直接导入普通Python wheel；保留经共同合成夹具对照的JS实现，或另建后端调用。不要为引入此包而直接改变现有网页实时采样流程。
