# MLQuant

Python 3.12 A 股因子与机器学习研究工具。提供公式化因子注册、点时数据校验、
离线研究报告、机器学习组合及独立的月／周／日频探索流程。

本仓库只包含代码、配置和使用说明。**研究结论、收益数字、模型排名、原始数据、
模型产物与交易信号均不发布，即使仓库为 private。** 见 [发布边界](docs/privacy.md)。

## 安装与使用

```shell
python -m venv .venv
# Activate the environment using your platform's activation script.
python -m pip install -e ".[dev]"
mlquant --help
mlquant status --root /path/to/local/data --json
mlquant factor list --json
```

仅支持 Python 3.12。可选 Tushare 导入工具使用 `pip install -e ".[data]"`。
通过 `--root` 或 shell 环境变量 `MLQUANT_DATA_ROOT` 指定本地数据根；`.env.example`
仅为模板，普通 CLI 不自动加载 `.env`。不要把令牌、数据湖或报告放进 Git。

## 功能与研究纪律

- 169 个注册因子及受限公式 DSL，支持版本、依赖与缓存。
- 不复权行情与独立累计后复权因子分开存储；财务按公告可得日处理。
- 月度正式报告与独立的日／周／月频探索；后者不能冒充实盘成交结果。
- 模型开发、验证、测试与监控分开，训练需要检查收益标签的结束时间。
- 行业分层/中性化要求真实点时行业及历史基准资料。代码能力不代表本地数据已合格；
  运行前必须审计，不得用快照、历史回填或未知分类替代正式输入。
- QMT 为可选模拟盘模板，默认不绑定报告，默认不下单。见 [QMT 说明](qmt/README.md)。

## 目录

|目录|内容|
|---|---|
|`src/mlquant/`|数据契约、因子、模型、报告、CLI 与本地 Web|
|`config/`|研究方法配置与示例，不包含实跑结果|
|`scripts/`|本地导入、审计、研究与静态报告工具|
|`tests/`|合成数据上的回归测试|
|`qmt/`|安全默认的模拟盘执行模板|
|`docs/`|使用、验证和发布边界|

`research/` 为本地私有笔记，`legacy/` 为本地历史工程档案，两者都不随发布上传。
数据和离线产物应放在独立数据根下。

## 验证

```shell
python -m ruff check src tests scripts
python -m pytest -q
python scripts/run_equity_smoke.py --output /path/to/local/smoke
```

更多命令见 [研究工作流](docs/workflows.md)。当前研究结论不收录在此 README。
