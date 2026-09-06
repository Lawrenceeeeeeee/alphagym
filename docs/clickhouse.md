# ClickHouse 存储、同步与迁移

ClickHouse 是行情、财务、因子缓存、运行记录及预计算报告的唯一持久化数据库。
Parquet/CSV 仅用于显式导入、导出。研究固定的是数据库内的版本号，不生成额外的
Parquet 快照。配置、后台进程日志、原始 QMT 文件和显式导出的交易信号仍是普通文件。

## 启动数据库

需要 Python 3.12，以及独立运行的 ClickHouse。仓库的 Compose 固定使用
ClickHouse 25.8.18.1。Windows 使用 Docker Desktop 的 Linux 容器；Linux/NAS
使用兼容的 Docker/Compose。数据库可以与研究程序分开部署。

在仓库根目录执行（PowerShell）：

```powershell
$env:MLQUANT_CLICKHOUSE_PASSWORD = "替换为自己的数据库密码"
docker compose up -d
$env:MLQUANT_CLICKHOUSE_USERNAME = "mlquant"
$env:MLQUANT_DATA_ROOT = "D:/mlquant-workspace"
python -m mlquant storage init --json
python -m mlquant factor sync --json
```

Linux/macOS 用 `export NAME=value` 设置同名环境变量。普通 CLI 不自动读取 `.env`。
Compose 默认仅监听本机；跨主机访问应配置 `MLQUANT_CLICKHOUSE_BIND`、防火墙与受控网络，
并在客户端设置实际主机。公网连接使用 TLS，不直接开放无保护的数据库端口。

`storage init` 在工作区生成非敏感的 `storage.yaml`，模板见
[`config/storage.example.yaml`](../config/storage.example.yaml)。支持：

- `MLQUANT_CLICKHOUSE_HOST`、`PORT`、`USERNAME`、`PASSWORD`、`DATABASE`、`SECURE`、`WORKSPACE`；环境变量覆盖 YAML。
- 密码只通过环境变量传入，禁止写入 YAML；Tushare Token 使用独立的 `TUSHARE_TOKEN`。
- `workspace` 是数据库内的稳定命名空间；换电脑时保留它。未保存配置时首次根据根路径生成标识。
- 一个数据库可以包含多个工作区；原生备份覆盖整个数据库。需要独立备份或权限边界时，使用独立数据库。

根目录现在主要承载连接配置与任务日志。查询和计算需要数据库在线，不再自动回退到旧文件。

## 导入已有数据

一次迁移旧数据根，包含 equity、factor_library、artifacts、models 中的受支持资源及
旧 SQLite 因子目录（迁移前停止旧程序写入）：

```shell
python -m mlquant storage migrate --source /path/to/old-data --root /path/to/new-workspace --json
```

新工作区先配置目标连接。迁移保留源文件，通过校验和与提交记录支持中断重试；
同名目标资源已有其他内容时拒绝覆盖。目标建议使用新的工作区。目录内的路径会转为
可迁移引用；指向源目录外的产物需要先整理到源目录内。源 SQLite 的活动写入/WAL
应由旧应用正常关闭后再迁移。

单表导入：

```shell
python -m mlquant equity-data import-parquet --root /path/to/workspace --input /path/to/daily.parquet --table daily --json
python -m mlquant equity-data import-parquet --root /path/to/workspace --input /path/to/adjustments.parquet --table adjustments --json
```

支持 daily、adjustments、calendar、securities、status、fundamentals、industries、index_members。
默认按业务主键更新，重复导入不会增加重复记录；`--mode replace` 显式替换当前版本。
大 Parquet 分批读取，QMT 日线分批解析后直接入库。导入失败时未发布的数据不可见。
QMT 一批日线与复权因子共同发布，保持“不复权原始价 + 独立累计后复权因子”。

## Tushare 同步

```powershell
python -m pip install ".[data]"
$env:TUSHARE_TOKEN = "自己的Token"
python -m mlquant equity-data sync-tushare --root "$env:MLQUANT_DATA_ROOT" --dataset market --start 2012-01-01 --json
python -m mlquant equity-data sync-tushare --root "$env:MLQUANT_DATA_ROOT" --dataset market --json
python -m mlquant equity-data sync-tushare --root "$env:MLQUANT_DATA_ROOT" --dataset fundamentals --symbol 600000.SH --json
```

也接受 `--token`，但环境变量可避免令牌留在命令历史中。市场同步按交易日分页、限速和重试，
首次默认从 2012 年开始，后续根据检查点回看 7 天，可用 `--overlap-days` 调整。
仅同步今天之前的完整日数据；不含实时行情订阅。日线、复权因子、检查点一起发布，
缺失对应复权因子的批次拒绝入库。`--no-daily-basic` 可关闭换手率和流通市值同步。

`--dataset` 支持 market、securities、fundamentals、all。财务按证券读取公告历史，
用实际公告日合并已经可得的字段，保留不同公告日的修订记录；为避免漏掉新证券及拆分接口
的较早字段，财务目前每次重取该证券历史后幂等入库，尚未实现财务接口的增量网络拉取。
`--start` 可限定输出的财务可得日期。Token 对应的接口权限、配额由用户账号决定。

不生成或回填历史行业、历史指数成分、ST/停牌等资料。正式中性化研究仍需导入真实
点时资料；量价因子的自动五分组只检查其所需输入。接口更新后的历史修订不会改变已经
固定版本的研究；但数据供应商未提供的历史修订不能由软件重建。

自定义来源可复用 `Workspace.write_table()` 或 `ClickHouseStore.write_frame()`；
多表同步用 `ClickHouseStore.batch()`，检查点与数据一并提交。当前内置提供 QMT、
Tushare 和 Parquet，尚未接入数字货币交易所的 WebSocket。

## 报告读取与导出

报告仍离线预计算，HTML 与数据存入 ClickHouse，Web 展示成品时不触发计算。
API 返回的 `path` 是兼容旧接口的逻辑资源名，`.parquet` 后缀不代表本地文件。
Python 使用 `Workspace.read_table()`/报告接口；要获得可直接打开或分享的文件：

```shell
python -m mlquant report export --root /path/to/workspace --report-id REPORT_ID --output /path/to/exported-report --json
```

导出目录包含 HTML、Markdown、Parquet、CSV 等成品，可以离线打开。
因子目录的 `factor export` 也显式导出普通 JSON 文件。

## 备份、恢复和 NAS

Compose 将数据与备份分别挂载到 `MLQUANT_CLICKHOUSE_DATA_DIR`、`MLQUANT_CLICKHOUSE_BACKUP_DIR`
（默认仓库下忽略提交的 clickhouse-data、clickhouse-backups）。服务器配置已允许
`backups` 磁盘。现有服务器需加入对应配置并重启；不能只设置客户端目录。

```shell
python -m mlquant storage backup --root /path/to/workspace --name full-20260906 --json
python -m mlquant storage backup --root /path/to/workspace --name inc-20260907 --base full-20260906 --json
```

备份在服务器端生成，覆盖所选数据库内所有工作区、版本和目录事件。将整个备份目录
复制到 NAS/新服务器的备份挂载点；增量恢复需要保留引用的完整基础备份链。
复制应用的 `storage.yaml`，保留 `workspace`，改 host/port 与目标 database，再设置密码。

目标数据库必须为空，例如 `mlquant_restored`：

```shell
python -m mlquant storage restore --root /path/to/new-workspace --name inc-20260907 --source-database mlquant --json
python -m mlquant status --root /path/to/new-workspace --json
```

迁移先使用相同 ClickHouse 版本，恢复并核对数据/报告后再升级。NAS 必须能运行兼容容器，
并具有足够内存与磁盘空间；仅有文件共享能力的 NAS 可以存备份，数据库运行在另一台主机。
跨主机/CPU 架构恢复应在目标机器实测；当前验证覆盖本机真实服务器的新数据库恢复。
不要直接复制运行中的 ClickHouse 数据目录来代替原生备份。

## 实现边界

量价、财务和研究表使用原生有类型的 ClickHouse 列；HTML/JSON 等成品作为资源内容存储。
行版本与提交清单共同决定可见性，失败的暂存批次不可见，旧版本保留用于复现。
自动回测和报告固定 equity 数据版本，并传给并行工作进程；不是另存一份文件快照。

现有因子目录 SQL 的约束和事务在**进程内临时 SQLite 内存库**执行，提交后的行事件
只持久化到 ClickHouse；不会创建 catalog.sqlite 文件。这是兼容现有关系型目录接口的
实现方式，并非 ClickHouse 支持通用关系型事务。目录会加载到进程内存，适合研究元数据，
不用于承载市场大表。

数据库级 DDL 锁串行发布数据/目录提交，备份排除锁表。进程崩溃后锁不会自动过期；
确认所有相关写入进程已退出、没有备份在执行后，由管理员检查并删除该数据库的
`data_publish_lock` 或 `catalog_write_lock` 残留表，再重试。不要在活跃写入期间删锁。
当前面向单服务器部署；多副本、自动故障切换、历史版本清理及孤立暂存批次回收尚未实现，
应监控磁盘容量。历史数据不自动删除。
