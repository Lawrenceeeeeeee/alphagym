# Python 库接口

还没有安装环境或数据？先按 [从零开始使用 MLQuant](getting-started.md) 完成准备。
本文代码中的 `workspace` 与数据路径沿用已初始化的本地工作区。

仅支持 Python 3.12。`pip install .` 安装数据、因子、基础合成和静态报告能力。
`pip install '.[web]'` 添加本地 Web；`.[ml]` 添加 sklearn 模型；`.[boosting]`
添加 XGBoost/LightGBM；`.[pdf]` 添加旧系列 PDF 导出；`.[data]` 添加 Tushare。
开发环境使用 `pip install -e '.[dev]'`，全功能使用 `.[all]`。

## 工作区

```python
from pathlib import Path
from mlquant import Workspace, WorkspaceConfig

workspace = Workspace(WorkspaceConfig(root=Path("/path/to/local/data")))
print(workspace.status())       # 不创建目录或数据库
workspace.initialize()         # 显式初始化，可重复调用
factors = workspace.list_factors(family="momentum")
print(factors[0]["factor_id"])
```

也可直接 `Workspace(Path(...))`，或 `Workspace()` 从 `MLQUANT_DATA_ROOT` 读取。
显式参数优先，不自动加载 `.env`，不退回当前目录。对象不持有常驻数据库连接；每个
方法负责打开和关闭连接。列表、状态和详情使用只读连接，不升级数据库。
旧目录升级先显式执行 `initialize()`。`list_factors()` 需要已初始化的因子库。
无数据根的内置定义发现使用 `mlquant factor list --json`。

## 注册和回测

```python
from mlquant import FactorDefinition

workspace.save_factor(FactorDefinition(
    factor_id="CUSTOM_RETURN_20D", name="CUSTOM_RETURN_20D",
    formula="=RETURN(market.adj_close, 20)",
    hypothesis_id="custom_momentum", family="momentum",
    expected_direction="positive",
))
run = workspace.run_factors(
    ["CUSTOM_RETURN_20D"], start_date="2014-01-01", end_date="2025-12-31",
    index_code="ALL_A", mode="formal",
)
print(workspace.run(run["run_id"]))
```

`run_factors()` 是同步、原始值五分组回测，按因子输入校验点时数据。量价因子需要
日线和独立累计后复权因子；财务、指数池分别增加可得日和历史成分要求。正式行业
中性化研究还要求历史申万一级行业与历史基准权重。缺失会抛错，不会回填快照。
`save_factor()` 不隐式启动缓存任务；需要缓存时使用 `mlquant.factor_cache.build_factor_cache`。

## 离线报告

```python
task = workspace.create_report(Path("/path/to/report.yaml"), ensure_runs=True)
output = workspace.execute_report(task["report_id"])
manifest = workspace.report_manifest(task["report_id"])
```

`create_report()` 接受 YAML/JSON 路径、字典或 `ReportSpec`，三种入口均经过同一校验。
创建后状态为 `queued`，不会自动执行。`ensure_runs=True` 会在创建前同步补足所需回测；
默认不会补算，执行报告时缺少覆盖区间的成功回测会失败。

后台执行可用 `workspace.start_report(report_id)`，然后
`workspace.wait_report(report_id, timeout=3600)`。后台进程使用当前 Python 环境中的
已安装包，与 CLI/Web 共用服务；进度与错误持久化到 ClickHouse，工作日志在
`factor_library/logs/`。`wait_report()` 只观察，不启动排队任务；超时抛 `TimeoutError`，
不终止后台计算。成功、失败和取消均返回终态，调用方应检查 `status`。

同一报告只允许一个执行者，完成或失败后不能重复执行；重新创建报告获得新 ID。
进程被操作系统强制终止时可能保留 `running`，当前没有自动重试或分布式调度器。
应检查本地日志后创建新任务。报告的读取、导出、Web 浏览均不触发重算。

## 更低层的扩展接口

|模块|用途|
|---|---|
|`mlquant.factors` / `mlquant.factors.base`|FactorRegistry、定义、计算上下文|
|`mlquant.factors.compute.compute_factors`|从日线、财务表与信号日期计算因子|
|`mlquant.research`|截面处理、分组和统计检验|
|`mlquant.pipeline` / `mlquant.account`|组合和成交记账|
|`mlquant.ml_composite`|冻结模型、隔离开发/验证/测试|
|`mlquant.ingest.import_qmt`|流式导入 QMT，不依赖 CLI，不自动重建缓存|
|`mlquant.factor_store.FactorStore`|高级目录管理，支持上下文管理器和 readonly=True|
|`mlquant.serialization.dumps`|严格 JSON，NaN/Infinity/缺失时间转 null|
|`mlquant.workflows.catalog`|发现、描述和运行九条可配置研究工作流|

Python API 保留原生异常：`DataContractError`、`ValueError`、`KeyError`、
`FileNotFoundError`、`TimeoutError`；缺少可选依赖抛 `OptionalDependencyError`，带安装提示。
库不调用 `sys.exit()`，也不通过捕获 CLI 输出返回研究结果。计算使用 DataFrame，大表持久化到 ClickHouse，
不塞进 JSON。模块的下划线成员属于内部实现，不应作为扩展点。

## ClickHouse 数据入口

先按 [部署指南](clickhouse.md) 设置数据库连接；`initialize()` 不负责启动数据库服务器。
`import_parquet(path, table=...)` 导入外部文件，`sync_tushare(token=..., dataset="market")`
同步市场数据。`read_table(name)` / `write_table(name, frame)` 访问标准数据表。
报告的 path 是逻辑资源名，普通 `Path.read_*` 不可读取；使用报告 API 或 CLI `report export`。
Parquet/CSV 文件只在显式导出时生成。
