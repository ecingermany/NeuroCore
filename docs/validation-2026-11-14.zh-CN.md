# NeuroCore 0.1.0 打包与验证记录

日期：2026-11-14（香港时间）。本轮实现Python包，不修改两个原工具仓库，不进行新的设备实验。

## 来源审计

- YYG当前main为b445e7d31b80abc2cfdad37fa5d9b562a4fc56b4。
- 只读Git比较确认feature_analysis.py、dynamic_txt.py、eeg_report与交接指定da014597054a01f95dc820210a663d85737aeb65无差异，工作区这些文件无修改。
- 原始文件SHA-256记录在provenance.json，同时随wheel携带。
- 提取器删除CLI、sys.path修改、报告/建议库自动加载；函数正文和数据类保留，内部import改为包内相对路径。
- EQ/PAF的业务建议文案改为宿主显式注入；回归使用原实现无库回退，详见接入说明。没有将建议库或个人数据装进包。
- NHS工作区参考dynamic-core.js已有未提交的mode5扩展（本次未修改）；金样对照使用其中1/4/2三种模式。两份JS参考文件的内容哈希记录在monitor-reference.json及安装包provenance.json，不将工作区内容冒充Git提交。

## 可公开复跑的验证

- 四份120秒合成记录，年龄6/30/65/12，种子20261114/20261115/20261116/20261117；最后一份增加高低频幅值和Theta下降以覆盖超标特征。
- 先由原YYG实现生成JSON金样，再对包内全局/逐秒频谱、显示增益、8类特征完整结果做逐项比较；容差rel=1e-9、abs=1e-10。
- 独立JS dynamic-core.js和dynamic-input.js生成183块合成数据基准，Python对照三轮继承历史的指标、阈值、条件、门控、奖励和计数。
- 额外检查窗函数幅值、30Hz增益边界、三槽阈值相位、非单调门控边界、未知门控、输入适配尾段、快照恢复、三项冷启动、年龄边界、游戏计分边界及可选Plotly/YYG HTML生成。
- 合成夹具不包含真实受测者数据，可随源码分发。

## 已有私有证据的本机回归

tools/validate_private.py仅读取既有材料，输出docs/private-validation-summary.json匿名摘要。

- 20组BFN/BAN：BRQ、SRQ、ATQ、ACQ、EQ、AntiStress、Synchrony、Symmetry、CQ、BQ、Rate对照。
- 补充LogBand、Slope、AbsoluteStrength、RelativeStrength及两类标签对照。
- 2026-11-11已知版本一次训练会话：三项P/W与BFN存档吻合；180个奖励与原独立验证脚本逐项一致。
- 未重启旧20组训练P/W跨版本拟合；没有声称新增屏幕逐格或硬件验证。
- 公开仓库不含本机私有输入，缺少材料时不能凭空重跑此脚本。

## 本次交付的限度

- 原工具还未切换import，未完成原工具GUI/报告全流程验收。
- BrainSee是默认无窗算法配置加通用渲染器，不是官方像素级复刻。
- BQtester实时3D路径尚未确认，不根据训练无窗强行推定。
- MeditationEngine为静态恢复；其他游戏只有已恢复的纯算法/状态消费，不包含完整Windows媒体、资源、题目和输入界面。
- MonitorR频谱是已验证指标重建口径，低频点与实际范围表达式的差异保留说明，不称物理标定。
- `.whl`和源码包通过本机构建及独立环境安装检查后交付；CI配置已提供，远端CI尚未运行。

实际测试数量、构建/安装结果见本次交付说明；环境约束见constraints-reference.txt。
