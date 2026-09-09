# 数字货币小时级合约研究

该模块研究 OKX 高流动性 USDT 线性永续合约，和秒级订单簿模块分离。首批只使用已确认的
OHLCV K 线，支持 1H、2H、4H。股票池要求合约在样本开始前至少已上市 365 天，按当前
24 小时折算 USDT 成交额初筛；后者仍有存活偏差，因此当前模块标记为探索研究。

```powershell
alphagym crypto-hourly --root <数据根> --bar 4H --universe-size 20 `
  --history-days 730 --horizon 1 --horizon 2 --horizon 6 `
  --fee-bps-per-side 5 --json
```

命令从公开接口读取行情，不需要交易凭据。K 线、特征、净值、指标、选择结果和 manifest
全部原子写入 ClickHouse 的 `factor_library/reports/crypto-<id>/`；不生成本地 Parquet/CSV。

## 首批假设

- 时间序列择时：单币多周期动量/反转、波动率、量能异常、区间位置与 K 线强弱。
- 横截面选币：同一时刻在币种间排序，做多顶部 20%、做空底部 20%，总绝对权重归一。
- 市场调整：用同期全市场中位数动量构造残差动量。
- 交互项：短期动量乘以成交额异常，检验放量确认。

信号使用本根收盘前可见信息，最早按下一根开盘成交。多根持有期按持有长度降采样，避免
重叠收益夸大有效样本。开发段确定 IC 方向；以开发、验证夏普的较小值排序；测试段只报告。
换仓按单腿收取设定的单边费用。

## 进入模拟盘前的门槛

当前 smoke 不构成上线依据。正式候选至少需要：两年以上历史；1H/2H/4H 频率比较；
2/5/10 bps 成本压力；点时股票池或预先固定股票池；资金费率、下架币与合约规则处理；
逐年和牛熊状态稳定性；冻结后新增数据前向验证。通过后才接 OKX 模拟盘，且先只生成信号，
再用限额仓位验证实际滑点、资金费率和下单失败率。

## 周频风格轮动

独立的 `crypto-style-rotation` 工作流把中期动量、短期反转、低波动、流动性和
资金费率 Carry 构造成市场中性的多空风格组合，再比较静态组合与 4/8/12 周因子动量配置：

```powershell
alphagym crypto-style-rotation --root <数据根> `
  --spec config/crypto_style_rotation.example.yaml --json
```

信号在周一 00:00 UTC 的 4H K 线收盘后形成，下一根 4H 开盘成交并持有一周。风格方向、
逆波动基准权重和轮动窗口只由 development/validation 决定；test 与 monitoring 只评估。
手续费按合并后的币种仓位换手计算，另计实际资金费率。

公开接口只能稳定发现当前存续合约。示例配置因此使用固定核心币，并将报告强制标记为
`PILOT`。只有输入确实包含历史下架合约和带版本的点时合约目录时，才可启用正式研究标记。

OKX 还公开提供多空账户比、合约持仓量/成交量和主动买卖量。先将它们持续增量入库：

```powershell
alphagym crypto-data import-okx-trading-statistics --root <数据根> `
  --period 1D --currency BTC --currency ETH --json
```

工作流据此生成 `risk_on`、`crowded_long`、`deleveraging` 和 `neutral` 状态，并在轮动报告中
做条件归因。接口当前返回的是有限近期窗口，因此数据未同时覆盖 development 与 validation
之前只可用于 monitoring/归因，不能参与模型选择。每日统计按次日才可用处理，防止日内回看。
