# 高频订单簿研究（探索模块）

AlphaGYM — Quant Research Toolbox 的独立高频研究路径，当前适配 OKX BTC-USDT
现货 400 档历史快照与增量消息。原有 A 股月频报告、行业中性化和 2026 年监控隔离不变。
这一路径不复用日频 `FactorContext` 的日期归一化，也不把秒级标签混进月度报告。

## 使用

仅使用 Python 3.12。在环境中设置数据库凭据，数据根使用 `--root` 或
`ALPHAGYM_DATA_ROOT`。示例研究划分见 `config/hf.example.yaml`。

```shell
alphagym hf list --json
alphagym hf build --root /path/to/data --start 2026-08-29 --end 2026-09-04 --json
alphagym hf run --root /path/to/data --spec config/hf.example.yaml --json
alphagym hf pipeline --root /path/to/data --spec config/hf.example.yaml --background --json
alphagym hf status --root /path/to/data --job-id <returned-id> --json
alphagym hf show --root /path/to/data --report-id <returned-id> --json
alphagym hf export --root /path/to/data --report-id <returned-id> --output /path/to/export --json
```

`build` 流式重建，每天原子发布；原始版本相同则复用样本。`run` 是显式同步计算命令。
`pipeline --background` 一次启动重建、研究和报告，`status` 读取持久化任务状态。
`workers` 控制盘口深度因子族的并行进程数。示例配置使用 8 个进程；设为 1 可复现单进程口径。

首轮因子组合使用已有特征报告作为冻结输入：

```powershell
alphagym hf combine --root /path/to/data --spec config/hf_ml.example.yaml --background --json
alphagym hf status --root /path/to/data --job-id <returned-id> --json
```

组合报告分别训练开发段的方向等权、IC 加权、Ridge、Elastic Net、PCA+Ridge、
Extra Trees 与直方图梯度提升。验证段选择方法，测试段只评估。每个预测周期独立拟合，
缺失值填充、标准化、标签剪尾和交易阈值均只使用开发段。监督模型分别预测未来
可成交多头报价收益与可成交空头报价收益，并在预测毛优势超过开发段冻结阈值时交易。
首轮成本网格重点覆盖 0–1 bps/边，同时保留 2 与 5 bps 压力情景。

## OKX 模拟盘

私有接口只通过 `OKXDemoClient` 访问，并固定发送 `x-simulated-trading: 1`。凭据只从
`OKX_API_KEY`、`OKX_SECRET_KEY`、`OKX_API_PASSPHRASE` 环境变量读取。先用非市价的
最小 `post_only` 订单完成下单与撤单闭环，再启动有明确时限的模拟盘会话：

```powershell
alphagym hf live --root /path/to/data --report-id <hfml-report> --horizon 5 `
  --duration-minutes 60 --execute-demo --json
alphagym hf live-status --root /path/to/data --session-id <session-id> --json
```

模拟盘会话使用冻结模型和冻结阈值，先积累 300 秒盘口历史。每次只允许一个仓位，
开仓后按报告持有期平仓；会话结束或运行异常时尝试关闭尚未平仓的模拟仓位。信号、
订单与状态写入 ClickHouse `factor_library/hf_live/<session-id>/`。
`show`、`export` 和 Web `/hf`、`/hf/reports/{id}` 只读预计算产物。
样本、报告和指标只存 ClickHouse；只有 `export` 输出本地文件。报告内嵌本地 ECharts，
无需互联网即可显示图表。未实现高频 Web 创建表单、后台任务调度和实盘执行。

## 第一批假设与文献

| 假设 | 实现 | 依据与解释 |
|---|---|---|
| 队列/深度不平衡 | 最佳 1/5/10/20 档买卖数量差除以总量 | [Gould、Bonart](https://arxiv.org/abs/1512.03492) 研究下一次价格变化方向；本模块另检验延迟后固定持有期收益 |
| 订单流不平衡 | 逐消息最佳档 OFI，回看 1/5/30 秒，按深度归一化 | [Cont、Kukanov、Stoikov](https://arxiv.org/abs/1011.6402) 研究供需变化与短期价格冲击；同期解释力不等于未来收益 |
| 多档深度结构 | 多档静态不平衡与第 20 档买卖距离差 | [Xu、Gould、Howison](https://arxiv.org/abs/1907.06230) 提供多档订单流研究依据；当前静态深度因子不是论文 MLOFI 的完整复现 |
| 加权中间价偏离 | 对侧数量加权的报价，相对 mid 的 bps 偏离 | [Stoikov](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694) 区分 weighted mid 与估计的 micro-price；当前实现仅 weighted mid，不冒充完整 micro-price 模型 |
| 短期动量/反转 | 过去 1/5/30 秒中间价收益 | 简单对照假设，正反方向只在开发段确定 |
| 不平衡变化 | 五档不平衡减去 5 秒前值 | 静态深度假设的变化量扩展，独立假设编号 |

首轮 162 个参数化因子以 `hf.*` 编号登记于 `FactorRegistry`；同类参数共享
`hypothesis_id`。实际特征由 `hf_factors.compute_features` 计算，注册计算器使用
保留毫秒时间的 `HFFactorContext`，禁止输入日频上下文。

完整候选地图、数据需求和后续实施顺序见[高频因子研究路线图](high-frequency-factor-roadmap.md)。

做市是另一类问题：[Avellaneda、Stoikov](https://www.tandfonline.com/doi/abs/10.1080/14697680701381228)
将报价决策与库存风险结合。仅有本次 L2 档案不足以可信重现挂单排队成交；
不使用“价格碰到报价就全部成交”的假设来制造做市收益。

## 数据与时间隔离

快照替换盘口；更新中的数量是绝对数量，零表示删除。所有消息按源记录连续顺序处理，
遇到倒序时间、缺失源行、非法价格/数量会报错。同毫秒消息一起处理；网格时点只读取
时间戳不晚于该时点的状态，不把整秒末值提前到秒初。超过 2 秒陈旧、空或交叉盘口不采样。
快照、缺秒、跨日会重置滚动窗口，首个不完整 OFI 区间保持缺失。

本批档案缺少 `seqId`、`prevSeqId`、checksum 和本地接收时间，源行连续不能证明交易所
消息无遗漏。规则参考 [OKX 官方市场数据说明](https://www.okx.com/docs-v5/trick_en/)。

开发段确定每个因子/持有期的方向（Rank IC 符号）与 20%/80% 阈值，验证与测试冻结。
报价标签从信号后至少 1 秒开始，持有 1/5/30/60 秒；固定 UTC 相位每“延迟+持有期”
采样一次，避免头寸重叠。入场与离场必须处于同一完整数据区段，禁止跨数据集划分。
主情景单边 5 bps 是研究假设，不是账户实收费率；同时列 0/1/2/5/10 bps。

候选门槛：开发和验证段都至少 100 笔、扣费均值都为正；按两段较小均值选一个。
没有候选通过就保留空结果。测试榜单全部公开供审计，但不能重新挑选或调整参数。
参数组合数写入 manifest，且不把高度相关变体当作独立发现。

## 如何解读结果

Rank IC 衡量信号与未来中间价收益的秩相关；收益诊断则使用实际买卖报价往返。
买入按 ask，卖出按 bid，另减两次假设手续费。双向收益含假设空头，不能视为无需
借币的现货策略；纯多头均值和笔数单列。双腿可见最佳档容量均需覆盖名义数量，
这仍是事后容量过滤的报价收益诊断，不是完整订单执行回测。当前没有排队、订单失败、
滑点冲击、资金占用、借币成本或组合净值模拟，也不年化一周的收益或夏普率。

一周内大量消息不等于大量独立样本：报告展示每日稳定性，不给独立同分布假设下的
显著性保证。收益不足以覆盖成本时，因子可作为后续执行研究的候选信息，但不能宣称
已找到可交易 alpha。后续试验需要预先冻结方案，再用新增日期验证。
