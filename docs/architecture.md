# 架构与发布准备

## 边界

```text
Python Workspace ─┬─ 应用服务 services ─ 领域计算 / ReportEngine
CLI              ─┤                        │
Web              ─┘                        ├─ FactorStore → ClickHouse 目录事件
                                           └─ 数据契约 → ClickHouse 数据列与预计算产物
后台进程 jobs ─ 已安装的 CLI worker ─ 同一 ReportEngine
```

领域计算仍使用原模块路径，避免破坏现有研究代码。`Workspace` 提供稳定的工作流入口；
共享报告校验与创建放在 `services.py`，本地进程管理放在 `jobs.py`。CLI 负责参数、
退出码和 JSON；Web 负责表单及成品展示。高级算法保留独立函数，可以直接组合和测试。

九条原脚本研究流程迁入 `alphagym.workflows`，每条通过显式 `Config` 和 `run()` 调用。
配置与数据根均由调用方提供，进度走 logger，结果返回状态及产物路径；原脚本保留薄包装。
白名单 catalog 为 Agent 提供列表、参数描述与配置分发，拒绝任意模块导入执行。

`WorkspaceConfig` 集中解析数据根。导入包不初始化目录、连接数据库或加载可选模型。
模型目录与模型训练分开，选一个模型时才创建对应估计器。基础等权/IC 衰减不依赖
sklearn。QMT 日线与复权因子分块暂存到 ClickHouse，单批提交清单共同发布；失败批次不可见。
研究读取固定数据库版本，Parquet 仅用于导入导出。因子目录 SQL 在临时内存 SQLite 中
执行约束，提交事件只持久化到 ClickHouse，不生成 SQLite 数据库文件。部署、锁恢复与
单机实现边界见 [数据库指南](clickhouse.md)。

报告严格使用本地随包 ECharts，HTML 内嵌数据。系列 PDF 是显式的可选功能；
`build_series(..., include_pdf=True)` 需要 `.[pdf]`，默认输出 Markdown、HTML 和数据文件。
历史 PNG 可以继续保留，但新代码不生成 PNG 图表。

## 兼容变更

- 旧模块导入和 `python -m alphagym.cli` 保留；新增 `python -m alphagym`。
- `report create --run` 改为后台启动，返回任务 ID；用 `report wait` 等待。
  需要同步时先 create，再 `report run --report-id ...`，或用 `Workspace.execute_report()`。
- 数据契约错误退出码从 2 修正为 1；命令用法错误保持 2。
- 基础安装不再带 Web、sklearn、boosting、PDF。旧开发环境使用 `.[dev]` 或 `.[all]`。
- 探索与指数补跑的 `--pool` 改为显式路径，不读取仓库未发布的私有默认因子池；
  旧 v0 脚本的 `--report` 也必须显式指定。
- 状态查询不再顺便初始化数据库；初始化使用 `factor sync` 或 `Workspace.initialize()`。
- `report build-series` 不再要求源码树中的默认配置文件。可显式提供 `--config`；
  没有输出路径时，必须指定数据根。

## 验证与发布

```shell
python -m ruff check src tests scripts
python -m pytest -q
python scripts/run_equity_smoke.py
python scripts/run_library_smoke.py --background
python -m build
```

`run_library_smoke.py` 仅使用合成数据，自动建立独立临时目录，覆盖 Python API、
因子回测、后台进程、等待和八类静态报告产物。CI 分别验证完整开发安装和仅基础 wheel
安装，后者在源码目录外运行，保证模板/静态文件随包且不依赖本机目录。
源码分发采用目录白名单，排除 research、真实数据和本地配置结果。

仓库公开和 PyPI 发布是另一个动作，目前没有自动发布步骤。发布前仍需仓库所有者
确定项目许可证，核对第三方 ECharts 的许可和随包声明，并对准备推送的实际历史做
凭据及私有产物审查。不要因为当前工作树干净就推送全部本地历史。见 [发布边界](privacy.md)。
