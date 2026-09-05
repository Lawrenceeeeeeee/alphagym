# 本地工作流

所有真实数据和结果留在本地。下列路径需替换成自己的数据根。

## 发现与审计

```shell
mlquant status --root /path/to/local/data --json
mlquant factor list --json
mlquant factor show MOMENTUM_60D --json
python scripts/audit_ml_reassessment.py --root /path/to/local/data --json
```

中性化研究数据不合格时，审计应阻断。不能通过改元数据标志来绕过检查。

## 报告与探索

```shell
mlquant report create --spec config/reports/smoke_ml_combo.yaml --run --json
mlquant report list --json
python scripts/evaluate_ml_exploratory.py --root /path/to/local/data --output /path/to/local/experiment --stage all
python scripts/report_ml_exploratory.py --output /path/to/local/experiment
```

正式报告当前持有期为 1M。无中性化的日／周频入口是独立探索流程，需要阅读生成的
metadata 中的执行、成本、缺价和样本隔离假设。静态报告在本地离线生成，不随点击重算。
研究结论与产物路径不得复制到仓库文档或提交。

## 验证与数据依赖

Ruff 和测试不应需要实际 API 凭据。`run_equity_smoke.py` 使用合成数据并把产物写到
明确的本地输出目录。真实数据导入是独立步骤，不能在 GitHub Actions 中自动拉取或上传。
