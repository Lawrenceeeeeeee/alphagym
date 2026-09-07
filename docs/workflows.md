# 本地工作流

首次使用请先阅读 [从零开始使用 AlphaGYM](getting-started.md)，完成 Python 环境、数据根
初始化和输入准备。本页是安装完成后的命令速查。

所有真实数据和结果保存在自己配置的 ClickHouse。下列路径需替换成自己的工作区根，
数据库启动与备份迁移见 [ClickHouse 指南](clickhouse.md)。

## 发现与审计

```shell
alphagym status --root /path/to/local/data --json
alphagym factor list --json
alphagym factor sync --root /path/to/local/data --json
alphagym factor show MOMENTUM_60D --root /path/to/local/data --json
python scripts/audit_ml_reassessment.py --root /path/to/local/data --json
```

中性化研究数据不合格时，审计应阻断。不能通过改元数据标志来绕过检查。

## 报告与探索

```shell
alphagym report create --root /path/to/local/data --spec /path/to/report.yaml --ensure-runs --run --json
alphagym report wait --root /path/to/local/data --report-id <returned-id> --json
alphagym report list --root /path/to/local/data --json
alphagym report export --root /path/to/local/data --report-id <returned-id> --output /path/to/export --json
python scripts/evaluate_ml_exploratory.py --root /path/to/local/data --pool /path/to/pool.yaml --output /path/to/local/experiment --stage all
python scripts/report_ml_exploratory.py --output /path/to/local/experiment
```

正式报告当前持有期为 1M。无中性化的日／周频入口是独立探索流程，需要阅读生成的
metadata 中的执行、成本、缺价和样本隔离假设。静态报告在本地离线生成，不随点击重算。
研究结论与产物路径不得复制到仓库文档或提交。

## 安装后的研究工作流

业务实现位于 `alphagym.workflows`，`scripts/` 中的同名入口只是兼容包装。
不需要复制源码脚本，不默认读取私有 `config/ml_pool.yaml`。

```python
from pathlib import Path
from alphagym.workflows.run_ml_exploration import Config, run

result = run(Config(
    root=Path("/path/to/local/data"), pool=Path("/path/to/pool.yaml"),
    models="ols,ridge", feature_mode="neutral", protocol="frozen",
))
```

因子池文件由调用方提供，格式为 `pool: [FACTOR_A, FACTOR_B]`，名称必须在本地因子库登记。
工作流有各自的数据与研究限制，配置构造会检查类型、选项和正数边界；缺失输入抛异常。
业务函数通过 `alphagym.workflows` logger 记录进度，不写标准输出、不退出宿主进程。

Agent 可用 `alphagym workflow list --json` 发现九条流程，用
`alphagym workflow describe <name> --json` 读取参数、类型、默认值与必填标志，再将配置
写成 YAML，通过 `alphagym workflow run <name> --config /path/to/workflow.yaml --json` 执行。
Python 对应 `alphagym.workflows.catalog.run_workflow(name, config)`；只允许注册的工作流名称。
工作流返回成功状态与产物路径，批量流程还返回分项状态。通用 workflow run 是同步入口。

## 验证与数据依赖

Ruff 和测试不应需要实际 API 凭据。`run_equity_smoke.py` 使用合成数据并把产物写到
明确的本地输出目录。真实数据导入是独立步骤，不能在 GitHub Actions 中自动拉取或上传。
