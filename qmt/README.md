# AlphaGYM QMT 模拟盘执行器

`paper_trader.py` 是国金 QMT 策略编辑器使用的内置 Python（innerApi）脚本，不是
MiniQMT 外部脚本。交易端不重新计算因子，只读取研究端冻结导出的：

- `signal_latest.csv`：证券代码、冻结模型分数、目标权重；
- `state.json`：报告/run/method、实际生效交易日、信号哈希，以及参与模型的每个
  `factor_id + revision_id + lookback_days`。

默认配置不绑定任何研究报告或模型。使用前必须同时修改
`SIGNAL_DIR`、`EXPECTED_REPORT_ID` 和 `EXPECTED_METHOD`，不能只替换 CSV。

## 使用顺序

1. 在仓库环境重新导出冻结信号：

   ```powershell
   python -m alphagym.cli signal export `
     --root D:\QuantData\lake `
     --report-id YOUR_LOCAL_REPORT_ID `
     --method YOUR_LOCAL_METHOD --top-n 50 --json
   ```

2. 将 `paper_trader.py` 粘贴到国金 QMT「策略交易」编辑器，选择**模拟资金账号**。
3. 保持 `DRY_RUN = True` 运行，检查信号目录生成的 `dry_run_plan.json`。
4. 只有在 `state.json.effective_trade_date` 当日，确认计划和账户后，才同时设置：

   ```python
   DRY_RUN = False
   SIMULATION_ACCOUNT_CONFIRMED = True
   ```

5. 运行时会生成 `execution_state.json`。卖单成交并释放资金后才进入买入阶段；有外部未完成
   委托、停牌/涨跌停、T+1 不可卖、拒单或部分成交时会 fail closed，保留 `blocked` 状态供人工
   核对，不会自动重复报单。只有目标持仓全部一致后才生成 `executed.json`。

旧版无 `schema_version=2`、`effective_trade_date` 或 `factor_manifest` 的信号会被拒绝，必须用
当前 `alphagym signal export` 重新导出。错过信号生效日时只允许 dry-run。
报告、信号、账户配置及执行产物均留在本地，不提交 Git。
