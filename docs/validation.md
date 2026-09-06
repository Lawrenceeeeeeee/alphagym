# 库化验证记录

此记录只描述软件验证，不包含真实市场研究或投资结论。

|目标|实现与验证证据|
|---|---|
|可安装的 Python 库|Python 3.12 wheel 与 sdist 构建成功，wheel 包含 py.typed、模板、本地 ECharts 和九条工作流|
|脱离源码目录|独立非 editable 环境在系统临时目录运行；导入路径确认来自 site-packages|
|基础依赖独立|独立环境未安装 sklearn、XGBoost、LightGBM、FastAPI、ReportLab 或 matplotlib；因子回测和后台报告仍通过|
|Python 复用|Workspace 注册、回测、创建/执行/等待/读取报告；九条工作流具有显式 Config 与 run 入口|
|Agent 调用|所有 CLI 叶命令接受 --json；测试覆盖参数位置、只读查询、领域错误、任务失败、工作流发现及配置拒绝|
|资源与任务管理|只读连接禁止写入；初始化幂等；报告原子占用阻止重复计算；QMT 空导入/中途失败不发布数据，保留旧资源|
|研究约束|原有点时、未来数据不变性、开发/验证/测试隔离、复权及成交回归测试保留并通过|
|静态产物|合成数据 smoke 验证 ClickHouse 中 spec、manifest、monthly、summary、correlation、combo、Markdown、HTML 八类报告资源，并验证显式导出；新产物无 PNG 图表|
|分发边界|检查实际 wheel/sdist 文件清单，未包含 research、legacy、本地数据表、数据库或缓存；源码分发采用白名单|

本地完整测试：183 项通过，Ruff 通过。测试有已有的模型迭代未收敛提示和 Web 测试依赖
弃用提示，均非测试失败。数据 smoke 和完整库/API 后台报告 smoke 均通过。

已添加带 ClickHouse 服务的 Linux 完整开发环境与基础 wheel CI，以及 Windows 打包 CI；远端 CI 尚未运行，
不能将本地验证等同于跨平台 CI 已通过。仓库公开、许可证选择和实际推送历史审查不在
本次本地执行动作中；没有发布 GitHub 或 PyPI。

## ClickHouse 验证（2026-09-06）

真实测试服务器：WSL Ubuntu 中 ClickHouse 25.8.18.1；客户端为 Windows Python 3.12。
测试数据使用独立临时工作区，不包含真实市场数据。验证覆盖：

- 分批导入、重复更新、空值/索引保留、筛选投影、模式扩展及流式读取。
- 发布前失败不可见、固定数据版本、晚提交不进入旧版本、并行因子缓存。
- Tushare 模拟接口的单位转换、分页、检查点与财务公告可得日；未使用真实 Token 联网下载。
- 旧 Parquet 与 SQLite 目录迁移、源文件保留、提交后重试及工作区路径迁移。
- 原生全量/增量备份分别恢复到新空数据库，并核对数据行和因子目录；未在真实 NAS 上执行恢复。
- 数据 smoke、后台报告 smoke、报告文件导出；仅基础依赖的独立 wheel 环境也完成后台报告。
- Compose 配置校验、wheel/sdist 构建及包内容检查。Docker 容器本身与远端 CI 尚未验证。

最后的小范围修正另做对应回归与命令验证；连接失败输出单一 JSON 领域错误，退出码 1。
正式工作区的数据搬迁还需要用户提供源数据根及目标连接信息；不能将合成数据验证视作
已迁移用户的真实数据。
