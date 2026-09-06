# 从零开始使用 MLQuant

这份指南假设你没有安装过 MLQuant，也没有项目原作者的数据、私有配置或运行环境。
先完成第 1～3 步，即可用合成数据跑出第一份报告；有真实数据后再继续第 4～7 步。

MLQuant 是研究工具：读取数据 → 计算因子 → 回测 → 离线生成报告。安装软件不会自动
获得行情、财务数据或数据源权限，也不会自动连接券商下单。

## 1. 准备 Python 和代码

### 安装 Python 3.12

从 [Python 官网](https://www.python.org/downloads/) 选择适合操作系统的 **Python 3.12**。
本项目要求 `>=3.12,<3.13`，不能直接使用系统自带的其他版本，也不要只点击最新版下载。
安装完成后重新打开终端：Windows 用 PowerShell，Mac 用“终端”。

Windows 检查：

```powershell
py -3.12 --version
```

Mac / Linux 检查：

```bash
python3.12 --version
```

应显示 `Python 3.12.x`。如果找不到命令，先检查 Python 是否安装、终端是否重开；
也可以使用 Python 3.12 解释器的完整路径。不要在版本不对时继续安装。

### 获取代码并进入项目目录

从项目提供方获取代码压缩包并解压，或在有仓库访问权限、已安装 Git 的情况下克隆：

```shell
git clone "REPOSITORY_URL" mlquant
cd mlquant
```

`REPOSITORY_URL` 必须替换成项目提供方给你的真实地址；本指南不假设项目已经公开。
下载压缩包的用户直接在解压后的 `mlquant` 目录打开终端即可。
后面的安装命令和 `scripts/...` 命令都在这个目录执行，应能看到 `README.md`、
`pyproject.toml`、`src/`。`pip install .` 中的点表示这个目录。

## 2. 建立独立环境并安装

Windows / PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

Mac / Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

如果 PowerShell 禁止执行激活脚本，不必修改系统策略；直接使用环境中的解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m mlquant --help
```

选择这种方式时，将本文后续命令开头的 `python` 换成 `.\.venv\Scripts\python.exe`。
虚拟环境的建立和激活方式见 [Python 官方 venv 文档](https://docs.python.org/3.12/library/venv.html)。

安装完成后检查：

```shell
python --version
python -c "import mlquant; print(mlquant.__version__); print(mlquant.__file__)"
python -m mlquant --help
python -m mlquant factor list --json
```

预期：Python 为 3.12，能够打印 MLQuant 版本和导入路径，最后一条命令输出内置因子列表。
这一步不需要数据目录。`python -m mlquant` 与 `mlquant` 等价，本文优先用前者，避免
命令不在 PATH 或误用其他 Python 环境。

### 按需安装功能

先用基础安装跑通第 3 步，需要额外功能时再安装。以下命令仍在代码目录执行。

|用途|安装命令|
|---|---|
|基础因子、基础合成、静态报告|`python -m pip install .`|
|本地 Web|`python -m pip install ".[web]"`|
|sklearn 机器学习模型|`python -m pip install ".[ml]"`|
|XGBoost / LightGBM，含 sklearn|`python -m pip install ".[boosting]"`|
|Tushare 导入|`python -m pip install ".[data]"`|
|旧系列报告的可选 PDF 导出|`python -m pip install ".[pdf]"`|
|上述功能一起安装|`python -m pip install ".[all]"`|
|修改代码、运行测试|`python -m pip install -e ".[dev]"`|

引号在 Mac shell 中也要保留。普通安装复制当前代码版本；修改源码或拉取更新后要重新
安装。开发安装 `-e` 会直接使用工作区源码。重新打开终端时需要重新激活环境。

## 3. 没有真实数据，也能跑出第一份报告

先按 [ClickHouse 部署指南](clickhouse.md) 启动数据库并设置连接环境变量。合成数据也写入独立的 ClickHouse 工作区。


在项目目录运行：

```shell
python scripts/run_equity_smoke.py
python scripts/run_library_smoke.py --background
```

第一条验证合成数据契约和基础报告输出。第二条自动完成：创建独立临时数据根、初始化
因子库、生成合成行情、注册测试因子、回测、启动后台报告、等待完成并检查产物。
它不需要 QMT、Tushare Token 或私有配置，也不会把合成数据写到你的正式数据根。

第二条成功时输出包含 `"ok": true`、`report_id`、`path` 的 JSON。
输出的 `path` 是 ClickHouse 逻辑位置，不是本地文件。用下面的命令导出，
再打开导出目录中的 `report.html`：

```shell
python -m mlquant report export --root <输出path中factor_library之前的根目录> --report-id <输出ID> --output <本地导出目录> --json
```


**此处的 SMOKE 报告仅验证软件流程，使用的是合成数据，不是投资研究结论。**
做到这里，就已经完成“从零安装并跑通框架”。没有真实数据时可以停在这一步。

## 4. 为自己的研究创建数据根

代码目录和数据根是两个不同的目录。例如代码在 `~/Projects/mlquant`，数据在
`~/mlquant-data`。不要把大型行情、数据库或研究报告放进 Git 仓库。

Windows / PowerShell：

```powershell
$env:MLQUANT_DATA_ROOT = Join-Path $env:USERPROFILE "mlquant-data"
New-Item -ItemType Directory -Force -Path $env:MLQUANT_DATA_ROOT | Out-Null
```

Mac / Linux：

```bash
export MLQUANT_DATA_ROOT="$HOME/mlquant-data"
mkdir -p "$MLQUANT_DATA_ROOT"
```

然后两种系统都运行：

```shell
python -m mlquant status --json
python -m mlquant storage init --json
python -m mlquant factor sync --json
python -m mlquant factor show MOMENTUM_60D --json
```

`status` 只检查现有内容，空目录是正常的；`factor sync` 才创建因子目录数据库并登记
内置因子，不下载行情。普通 CLI 不自动加载 `.env`，仅复制 `.env.example` 不会生效。
上面的环境变量只作用于当前终端会话；每次重新打开终端都要重新设置，或自行加入 shell
启动配置。也可以每条命令显式传 `--root "你的数据根绝对路径"`，它优先于环境变量。

工作区根保存 `storage.yaml` 与 `factor_library/logs/`；行情、目录、报告与缓存
保存在 ClickHouse。兼容 API 的 `equity/*.parquet` 等路径是数据库逻辑资源名。

## 5. 准备真实数据：三种入口任选其一

### A. 已有 MLQuant 数据

使用显式迁移命令，保留源文件：

```shell
python -m mlquant storage migrate --source <旧数据根> --root <已配置的新工作区> --json
```

详见 [迁移与备份](clickhouse.md)，不要直接复制文件后期待自动读取。

### B. 已有 QMT 本地行情文件

QMT 来源目录应包含 `SH/86400/`、`SZ/86400/` 等日线目录，以及独立的 `DividData/`。
把下列路径替换为你的实际目录。Windows 示例：

```powershell
python -m mlquant equity-data import-qmt --datadir "D:\QMT\datadir" --no-refresh-factor-cache --json
```

Mac 上若已复制原始文件，可用同一个解析器：

```bash
python -m mlquant equity-data import-qmt --datadir "$HOME/qmt-datadir" --no-refresh-factor-cache --json
```

这里使用第 4 步配置的数据根，首次导入显式关闭后台缓存重建，避免顺带启动全因子缓存。
解析器不依赖 xtquant，但复制行情文件不等于在 Mac 上安装或运行了 QMT 客户端。

导入直接写入 ClickHouse 日线与独立复权因子，不生成 Parquet 文件。原始 `.DAT` 不包含
复权因子；不要漏掉 `DividData`。这也不会补出财务、行业、指数成分等其他资料。
导入会更新这两个文件，已有数据先备份，导入期间暂停研究任务。

### C. 自行接入其他数据源

库支持标准化的 Parquet 输入，通过 `equity-data import-parquet --input <文件> --table <表名> --root <工作区>` 导入数据库。
下表的文件名仅代表对应数据表；股票代码统一使用
如 `600000.SH`、`000001.SZ` 的形式，日期字段保存成可被 pandas 读取的日期，
字段单位在不同数据批次之间保持一致。

|文件|基本字段与用途|
|---|---|
|`daily.parquet`|`trade_date, symbol, open, high, low, close, volume, amount`；其他因子可能还需 turnover、float_market_cap 等|
|`adjustments.parquet`|`trade_date, symbol, adjust_factor`，累计后复权因子|
|`calendar.parquet`|`trade_date, is_open`，交易日历|
|`securities.parquet`|`symbol, list_date, delist_date`，上市与退市日期|
|`status.parquet`|`trade_date, symbol, is_st, is_pt, is_suspended, limit_up, limit_down`|
|`fundamentals.parquet`|`symbol, stat_date, available_date`，以及因子使用的财务字段|
|`industries.parquet`|`symbol, industry_code, industry_name, valid_from, valid_to, source, version`|
|`index_members.parquet`|`index_code, symbol, valid_from, valid_to, benchmark_weight`|
|`metadata.json`|来源、单位、快照/补值限制等真实说明|

这张表是基本数据契约，不是“只建空表就能研究”。行情 OHLC 保留不复权价格；使用时
按 `adj_close = close × adjust_factor` 计算后复权。累计因子仅除权日有记录时，按股票
向前填充，首个除权事件前为 1.0。不得用随未来事件改写历史的前复权数据替代。
财务可得日不能用财报期末日代替，历史成分不能用当前成分回填。

可选 Tushare 导入器的参数可查看：

```shell
python -m pip install ".[data]"
python -m mlquant equity-data sync-tushare --help
```

先在当前 shell 设置自己的 `TUSHARE_TOKEN`，不要写入仓库。量价部分的调用示例：

Windows：

```powershell
$env:TUSHARE_TOKEN = "替换为你自己的Token"
python -m mlquant equity-data sync-tushare --root "$env:MLQUANT_DATA_ROOT" --start 2012-01-01 --end 2025-12-31 --dataset market
```

Mac / Linux：

```bash
export TUSHARE_TOKEN="替换为你自己的Token"
python -m mlquant equity-data sync-tushare --root "$MLQUANT_DATA_ROOT" --start 2012-01-01 --end 2025-12-31 --dataset market
```

数据接口需要自己的权限与配额。市场数据支持检查点续传与重复更新；财务保留公告可得日。
不会补造历史行业、成分或 ST 状态，正式中性化研究仍需自行提供真实点时资料。
完整选项、Token 配置与同步边界见 [数据库指南](clickhouse.md)。

### 怎么判断数据够不够？

先运行 `python -m mlquant status --json` 看文件和字段是否存在。它不是研究资格认证。

- ALL_A 原始值五分组的量价因子：至少需要日线和复权因子；具体因子还可能要求扩展行情字段。
- 财务因子：增加财务字段及真实可得日。
- 指数股票池：增加历史成分。
- 正式行业分层/中性化：需要真实历史申万一级行业、历史指数成分与权重及其他完整输入。

完整数据契约和正式行业研究的审计入口：

```shell
python -m mlquant equity-data audit --index 000300.SH --json
```

它会检查八张表；只导入量价数据时可能报缺表。不要把它与原始值量价回测的按需校验
混为一谈。第 6 步的因子回测会按所选因子的输入单独检查，缺少必需数据会明确失败。

## 6. 用自己的数据生成第一份报告

先确保当前终端已激活环境并设置数据根，且执行过 `factor sync`。
打开仓库的 [报告示例](../config/report.example.yaml)，了解并按实际数据修改：

- `mode: smoke` 是带水印的流程验证；正式研究使用 `formal` 并满足所需点时条件。
- `universe.index_code: ALL_A` 是全 A 股池；指数池需要历史成分。
- `window` 是研究区间；示例为 2014～2025，数据应覆盖它并预留因子回看历史和交易日。
- `splits` 的 development / validation / test 必须首尾相接且覆盖整个 window。
- `holding_period: 1M` 是持有期；因子自身的回看天数是另一个概念，会进入 manifest。
- `factors.include` 指定已注册因子；示例是 `MOMENTUM_60D`。

不能把区间简单改短而保留原来的 splits；测试期不能用来调整方向和参数，2026 只作
独立监控。首次用示例时可以直接执行下面的命令；定制 spec 建议另存到自己的数据根，
然后把 `--spec` 改为该文件路径。

```shell
python -m mlquant report create --spec config/report.example.yaml --ensure-runs --run --json
```

`--ensure-runs` 会先同步补跑报告所缺的因子回测，数据量大时这一步可能较久；
`--run` 随后启动后台报告并返回任务 ID。它不是“运行一次 status 就自动开始计算”。
如果希望先手动验证单因子，可在创建报告前运行：

```shell
python -m mlquant factor run --factor MOMENTUM_60D --index ALL_A --start-date 2014-01-01 --end-date 2025-12-31 --mode smoke --json
```

复制创建报告输出中的 `report_id`，把下列命令中的占位文字替换成真实 ID：

```shell
python -m mlquant report status --report-id "替换为返回的report_id" --json
python -m mlquant report wait --report-id "替换为返回的report_id" --timeout 3600 --json
python -m mlquant report show --report-id "替换为返回的report_id" --json
```

`wait` 成功应返回 `status: succeeded`；失败时读 `error`，必要时查看数据根下
`factor_library/logs/` 的日志。超时只结束等待，不会杀掉后台任务，稍后可再次查询。
不要重复启动同一报告；失败后修正数据或配置，再创建新报告。

## 7. 阅读结果、使用 Web

成功后的 `path` 是 ClickHouse 逻辑位置 `{数据根}/factor_library/reports/{report_id}/`。先用 `report export` 导出，得到：

|文件|用途|
|---|---|
|`report.html` / `report.md`|浏览器或文本阅读|
|`spec.yaml`|本次研究配置|
|`manifest.json`|因子、回看期与产物记录|
|`monthly.parquet` / `summary.csv`|月度明细与汇总|
|`correlation.parquet` / `combo.json`|相关性及配置所需的合成结果|

直接打开 `report.html` 即可。配置未启用合成时，不要把对应空结果误认为有模型表现。
报告是离线成品，浏览不会重新计算。查询已有报告用：

```shell
python -m mlquant report list --json
```

如果希望使用本地网页界面，在代码目录安装 Web 额外依赖后启动：

```shell
python -m pip install ".[web]"
python -m mlquant factor serve --port 8765
```

浏览器访问 `http://127.0.0.1:8765`，报告列表在 `/reports`，新建表单在 `/reports/new`。
终端保持运行；停止 Web 用 Ctrl+C。网页创建报告同样需要已有数据和成功回测，不会自动
下载数据。仅打开静态 HTML 不需要运行这个服务。

## 8. 从 Python 或 Agent 调用

在激活后的环境中，将下面代码存为你自己的 Python 文件运行；数据根与前面相同。
此示例会真实执行研究，所以先完成数据准备。

```python
from pathlib import Path
from mlquant import Workspace

workspace = Workspace()  # 从 MLQUANT_DATA_ROOT 读取
workspace.initialize()
print(workspace.status())
task = workspace.create_report(Path("config/report.example.yaml"), ensure_runs=True)
workspace.start_report(task["report_id"])
result = workspace.wait_report(task["report_id"], timeout=3600)
if result["status"] != "succeeded":
    raise RuntimeError(result["error"])
print(result["path"])
```

这里的 spec 相对路径要求从项目目录执行；其他程序使用时改为自己 spec 的绝对路径。
如果不需要后台进程，用 `workspace.execute_report(task["report_id"])` 同步执行，
不要对同一 ID 同时调用两种执行方式。更多接口见 [Python API](python-api.md)。

Agent 的建议顺序是：status → factor list/show → 写 spec → report create → wait → 读成品。
主 CLI 的 `--json` 输出供程序读取，退出码 0 表示成功、1 表示领域错误、2 表示命令用法错误；
失败信息可能在 stderr，调用方应同时保留 stdout、stderr 和退出码。

其他研究流程可先发现再调用：

```shell
python -m mlquant workflow list --json
python -m mlquant workflow describe ml-exploration --json
```

根据返回的必填字段、类型和默认值编写自己的 YAML 配置，再使用
`python -m mlquant workflow run ml-exploration --config "自己的配置文件.yaml" --json`。
不要直接运行占位配置；不同流程的数据、模型依赖不同，详见 [研究工作流](workflows.md)。

## 9. 下次使用、更新和常见问题

下次使用只需：进入代码目录 → 激活 `.venv` → 设置 `MLQUANT_DATA_ROOT` → 查询/运行。
无需重复下载全部行情。更新代码后重新运行对应的 `python -m pip install ...`；升级已有
数据根前先备份，再执行 `factor sync`。不要在旧版本后台任务仍运行时覆盖安装。

|现象|检查方法|
|---|---|
|找不到 Python / 版本不符|回到第 1 步，明确使用 Python 3.12|
|`No module named mlquant`|确认激活了安装时的环境；用 `python -m pip show mlquant` 查看|
|找不到 `mlquant` 命令|改用 `python -m mlquant`，检查环境是否激活|
|提示缺少数据根|当前终端重新设置环境变量，或显式传 `--root`|
|Catalog missing / 找不到因子|对同一数据根执行 `factor sync`；检查是否写错根路径|
|缺少 sklearn / XGBoost 等|回第 2 步安装对应额外依赖；Mac 上先确认基础 smoke 通过，再排查该依赖|
|缺表、缺字段、缺成功回测|按第 5 步准备数据；报告创建可加 `--ensure-runs`，但它不会下载行情|
|全量审计报行业/状态表为空|它检查完整契约；量价五分组走对应因子回测的按需校验|
|报告显示 queued / running|查看 status 和工作日志；wait 超时不会取消任务|
|进程意外退出后一直 running|当前没有自动恢复调度；确认进程已结束后检查日志、修正原因，再创建新任务|
|文件存在但历史报告读不到|检查元数据中的绝对路径，尤其是跨机器迁移后|

开发者验证（先安装 `.[dev]`）：

```shell
python -m ruff check src tests scripts
python -m pytest -q
python scripts/run_equity_smoke.py
python scripts/run_library_smoke.py --background
```

## 10. 换电脑或迁移到 NAS

代码与 Python 3.12 虚拟环境在新机器重新安装。复制非敏感的 `storage.yaml`，保留 workspace，
设置数据库密码后即可连接原来的 ClickHouse；不需要复制本地行情文件。

同时搬迁服务器时，使用 `storage backup` 创建原生备份，将完整备份链复制到目标服务器，
再用 `storage restore` 恢复到空数据库。步骤与 NAS 配置见 [数据库指南](clickhouse.md)。
原始 QMT 数据、私有 spec、笔记、后台日志及显式导出文件需单独保留。

旧版 Parquet/SQLite 工作区先用 `storage migrate` 导入，源文件保留。新的目录索引使用
可迁移的资源引用；外部模型依赖和券商客户端仍需在目标机器单独配置与验证。
