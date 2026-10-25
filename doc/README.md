# OneFLOW 文档索引

`doc/` 是 OneFLOW 的工程文档根目录。页面按**生命周期**分区：长期维护的活文档、
一次性快照、可复核证据、正式报告互不混放。

> 文档命名遵循 [DOCUMENT_NAMING.md](DOCUMENT_NAMING.md)。新增文档先选目录与
> artifact，再写内容；删除或合并旧文档必须同步更新本索引和所有引用。

## 1. 目录与保留策略

| 目录 | 放什么 | 保留策略 |
|---|---|---|
| [`architecture/`](architecture/) | 稳定的架构说明、边界审计 | 每个主题只保留**一份**现行版本；被取代的审计直接删除（git 历史即档案） |
| [`plans/`](plans/) | 活文档：路线图、长周期计划 | 只放**living**文档（无日期后缀）；一次性 checklist 完成后删除 |
| [`handoff/`](handoff/) | 会话交接 | **至多一份**现行 handoff；新交接提交时删除上一份 |
| [`runbooks/`](runbooks/) | 构建、集群、移植等操作流程 | 每个流程一份；步骤变更就地更新，不建副本 |
| [`evidence/`](evidence/) | 可复核的环境/构建事实 | 与某一轮结论绑定的证据；结论被取代后随之删除 |
| [`reports/`](reports/) | 正式报告（performance / regression / delivery） | **不删已发布的报告**；过时报告保留但标注 historical，Markdown 与 HTML 同 stem 同步 |
| [`source/`](source/) | Sphinx 教程源（Read the Docs） | 与工程文档无关，按教程需要维护 |
| `_drafts/` | 未定稿草稿 | 定稿后删除，不长期保留 |

## 2. 现行入口（从这里开始读）

| 想了解 | 文档 |
|---|---|
| 当前状态、A–G 路线图与勾选进度、每轮门禁 | [plans/oneflow-development-todo.md](plans/oneflow-development-todo.md) |
| 长期阶段的取舍、里程碑与开放问题 | [plans/oneflow-development-plan.md](plans/oneflow-development-plan.md) |
| 上一次会话交接（可直接贴给新会话） | [handoff/oneflow-3d-hip-handoff-20261025.md](handoff/oneflow-3d-hip-handoff-20261025.md) |
| 昆山工作区固定路径、作业规范、清理规则 | [runbooks/oneflow-kunshan-workspace-runbook.md](runbooks/oneflow-kunshan-workspace-runbook.md) |
| DCU 移植方法论（契约→oracle→门禁→benchmark） | [runbooks/oneflow-dcu-porting-playbook-20260903.md](runbooks/oneflow-dcu-porting-playbook-20260903.md) |
| 主 solver 加速器架构审计与接入边界 | [architecture/oneflow-1d-euler-dcu-architecture-audit-20260904.md](architecture/oneflow-1d-euler-dcu-architecture-audit-20260904.md) |
| 报告列表 | [reports/README.md](reports/README.md) |

## 3. 仓库内与上游共有的文件

`doc/` 里既有 fork 维护的文档，也有 upstream 拥有的文档。动手前的判断规则：

- **可自由整理（fork 维护）**：`plans/oneflow-development-*.md`、`handoff/`、
  本索引、`DOCUMENT_NAMING.md` 中 fork 新增的章节。
- **与 upstream 共有**：`architecture/`、`evidence/`、`reports/`、`runbooks/` 下的
  报告与流程、`source/`。可以更新内容，但不要整体移动或重写，否则上游同步时会产生
  结构性冲突；确需移动时单独提交并同步修复 `ports/`、`ci/` 等处的引用。
- `doc/reports/architecture/` 只存在于 upstream（fork 已按目录规范拆到
  `plans/`、`runbooks/`、`reports/delivery/`）；不要在本仓库重新创建该目录。

## 4. 构建 HTML 文档

```bash
cd doc
make html     # 输出到 doc/build/html/，CI 与 Read the Docs 使用 doc/source/
```

正式报告若同时提交 HTML，必须与 Markdown 使用相同 stem（例如
`oneflow-euler-performance-current.md` / `.html`），并在同一提交中更新。

## 5. 每轮收尾检查表

1. `plans/oneflow-development-todo.md` 的勾选、状态快照与本轮证据已更新。
2. 新交接提交后，旧的 `handoff/*` 已删除；一次性 checklist/草稿已删除。
3. 新增/删除/改名文档后，本索引、`doc/reports/README.md`、`AGENTS.md` 的链接
   全部有效（`grep` 一遍旧文件名应无残留）。
4. `git diff --check` 通过；文档中不含原始 CI/Slurm 日志、主机名、账号、作业号、
   绝对私有路径。
