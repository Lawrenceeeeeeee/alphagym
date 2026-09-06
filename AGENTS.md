# MLQuant A股开发规范

- 仅支持 Python 3.12；业务路径必须使用 `pathlib.Path`，数据根目录来自配置、`--root` 或
  `MLQUANT_DATA_ROOT`。
- 行业分层/中性化的正式研究必须使用点时申万一级行业、指数历史成分与权重、财务可得日；缺失即阻断。
- 因子库自动回测（原始值五分组，不做行业分层与中性化）按因子所需输入做点时校验：量价因子只需
  日线+复权因子，财务字段需可得日，指数池需历史成分；不要求历史申万行业与官方权重。
- 当前行业/成分快照只能用于带水印的 smoke 产物，严禁回填历史。
- 信号月末收盘后形成，下一交易日开盘成交；未来数据不得改变历史因子、组合或交易。
- 复用逻辑放 `src/mlquant/`；定义与配置放 `research/`、`config/`；大数据与产物不得提交。
- 正式因子必须登记在 `FactorRegistry`，缺失保持 NaN，具体参数共享稳定 hypothesis_id。
- 截面顺序固定：股票池、5×MAD、标准化、行业+对数流通市值 WLS、中性残差标准化。
- 组合在行业内五层，五组行业权重匹配基准；行业内等权且边界权重可拆分。
- 2026 数据不参与首期选模。开发/验证/测试必须分栏，测试期不可调方向或参数。
- 修改后运行 `python -m ruff check src tests scripts`、`python -m pytest -q` 和数据/报告烟雾测试。
- ClickHouse 是数据、缓存、运行目录和报告的唯一持久化数据库；Parquet/CSV 仅显式导入导出。
  测试需要独立 ClickHouse 服务。`root/storage.yaml` 保存非敏感连接与稳定 workspace；密码、
  Tushare Token 从环境读取。使用 `storage_io`/`ClickHouseStore` 访问逻辑资源，不直接读写文件。

## 报告区（研究报告）

- 研究报告全部离线预计算并保存为 ClickHouse 资源（逻辑位置 `{root}/factor_library/reports/{report_id}/`：
  spec.yaml、manifest.json、monthly.parquet、summary.csv、correlation.parquet、combo.json、
  report.md、report.html），Web 与 CLI 只读成品，点击不触发计算。`report export` 才输出本地文件。
- 报告 spec 必须写明：股票池（指数/点位申万行业/代码清单）、研究区间、开发/验证/测试三段
  划分（首尾相接、覆盖区间）、**持有期**（当前仅 1M：月末信号、次日开盘成交、持有至下月信号）
  与**因子回看期**（各因子 `lookback_days` 进单因子表与 manifest）。
- 合成对比严格遵守隔离：方向符号、去相关筛选、权重与方法只在 development(+validation) 确定
  （截至验证期末冻结），test 仅评估；monitoring（2026）只贴标不选模。
- 报告创建双入口：CLI `mlquant report create --spec x.yaml [--run]`（agent 首选）与 Web
  `/reports/new` 表单；后台执行、`mlquant report wait` 轮询、进度写入 `report` 表。
- 图表一律用前端 ECharts（`static/echarts.min.js` 本地随包）渲染，数据以 JSON 内嵌；
  **禁止用 matplotlib 生成 PNG 图表**（旧 PNG 产物仅作兼容保留，不再生成）。

## Agent 接口约定

- 所有命令支持 `--json` 输出稳定 JSON 结构；领域错误输出
  `{"ok": false, "error": {"code", "message"}}` 并退出码 1，用法错误退出码 2。
- 三条标准工作流：
  1. 体检：`mlquant status --root <root> --json`（数据库表行数/数据版本/进行中任务）；
  2. 发现：`mlquant factor list --json`、`mlquant factor show <id> --json`、
     `mlquant run show --run-id <id> --json`、`mlquant report list --json`；
  3. 研究：写 report spec YAML → `mlquant report create --spec x.yaml --run --json`
     （或 create + `report wait`）→ API 读取报告资源，或 `report export` 后读取本地成品。
- 因子指标筛选（spec `factors.metrics`）基于各因子最近成功 run 的 `factor_metric` 表
  （ALL_A、raw、base_5bps 口径）；排序默认按 |Rank IC|。

## QMT 数据与复权

- QMT 日线 `.dat`（`{datadir}/{SH|SZ|BJ}/86400/{代码}.DAT`，8 字节头 + 64 字节定长记录）只存
  不复权的 OHLCV、成交量、成交额，**不含复权因子**；价格按整数 ×1000 存储。
- 复权因子在 `{datadir}/DividData` 的 LevelDB（`.ldb`）中，独立于日线。key 为
  `市场|代码|4000|除权日毫秒`（+8 字节序号），value 96 字节：`bytes 8..15` 是除权日时间戳、
  `bytes 72..79`（float64）是单次除权的后复权因子。解析器在 `src/mlquant/qmt_dividend.py`
  （纯 Python，无 xtquant 依赖）。
- 回测只用后复权（`adj_close = close × 累计后复权因子`），禁用前复权：前复权会随未来除权改写
  历史价格，破坏点时性。存储形态保持「不复权原始价 + 独立复权因子」，需要时现算。
- `adjust_factor` 是累计后复权因子，仅在除权日有记录（该日及之后生效），除权前为 1.0；使用时
  `ffill().fillna(1.0)`。
- 入库命令：`python -m mlquant.cli equity-data import-qmt --datadir <QMT datadir> --root <数据根>`，
  写入 ClickHouse 逻辑资源 `equity/daily.parquet` 与 `equity/adjustments.parquet`（日线分块流式写，
  同批发布，避免一次性 concat 全市场爆内存；不生成 Parquet 文件）。
- 全量日线在模拟盘客户端（如 `D:\国金QMT交易端模拟\datadir`），实盘客户端通常只下载跟踪标的。
  DividData 混有 `.HK`/基金/债券/BJ，正式研究按 6 位 A 股代码过滤；目录里的 `000001_9000.DAT`
  等非 6 位代码文件及坏记录在导入时跳过。
