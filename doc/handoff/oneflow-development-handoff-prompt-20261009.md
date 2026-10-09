# OneFLOW development handoff prompt

**Date:** 2026-10-09  
**Purpose:** Copy the prompt below into a new coding-agent session to continue the current work.

## Prompt

```text
你接手的是 OneFLOW 仓库。请先阅读仓库根目录 AGENTS.md、dev 分支上的
doc/plans/oneflow-development-todo.md、ci/kunshan/README.md，以及本文件的
“当前状态”。以 git 和 GitHub 的实时状态为准；文档中的带日期记录是历史证据，
不能自动当作当前状态。

先做以下只读核对：
1. git status、当前分支、master/dev/topic 分支与 origin/upstream 的 ahead/behind。
2. 刷新 upstream PR #203 的 CI、review、mergeability 与最新提交：
   https://github.com/eric2003/OneFLOW/pull/203
3. 阅读开发 TODO 的 P0、P2.6、P2.7 和 P2.8，识别已完成、未完成和相互矛盾的条目。

当前优先事项是收口 PR #203。该 PR 修复 DataBaseType::GetIndex/GetName 首次访问时
静态类型映射可能尚未初始化的问题，并新增独立进程冷启动回归测试。Kunshan OpenAPI
已用 GCC 16.2.0、OpenMPI 5.0.11、CMake 4.4.3 运行该测试，结果 1/1 通过。最近一次
记录的 PR 状态是 MERGEABLE、等待 review；deploy 已成功，Linux 和 Windows CI 仍在运行。
这些状态会变化，必须先重新查询。

PR #203 分支是 pr/database-type-map-init，已推送到 origin。dev 也含有对应的生产代码
修复；tests/database 的新冷启动测试属于 PR topic。不要把 dev 上的 fork-only 计划、
handoff 或其他未发布文档带入 upstream PR。不要在本机编译；需要测试时使用 Kunshan，
优先 scnet-hpc OpenAPI backend，并遵循 ci/kunshan/README.md 的资源与证据要求。

PR 后续操作：
- 若 CI 失败或 review 提出问题，先复现并查明原因，再在 PR topic 分支修复、提交、推送，
  并重新验证。未经验证不要声称 PR 完成。
- 若 CI 全绿且 review 已处理，更新 dev 上的 TODO，记录准确结果；按需等待上游合并，
  合并后再同步 master/dev、清理 topic branch，并核对 origin/upstream。
- 对 MPI rank-local 初始化行为，如现有测试证据不足，补充针对真实数据库广播路径的
  Kunshan CPU MPI 验证；区分单元测试通过与 MPI 集成验证通过。

PR 收口后再推进长期路线：
- 当前受限 HIP 主路径为单 zone、finest grid、5 方程、Lax-Friedrichs、limiter off、
  inviscid。已有 gradient → reconstruction → flux/residual → state-update 的目标节点
  accuracy/stability 验收；这不代表完整 MPI/halo、多 zone、viscous/turbulence 或完整
  GPU residency 已完成。
- P2.6 主路径已完成，但 interface、periodic、solid-surface bc_q、ordinary boundary、
  ghost gradient copy、generation/token cache invalidation 等 contract 仍有待办。
- TODO 的性能记录显示 HIP 端到端主要瓶颈在 host CALC_BOUNDARY / NsCalcBc。下一步应先
  核实 boundary/ghost 的实际调用链和 CPU oracle 语义，再决定小 case contract 与设备迁移
  顺序；不要继续优化已经不是主要瓶颈的 kernel，也不要把探索性单作业 timing 写成稳定
  加速结论。
- 每次只推进一个边界清楚的切片，保留 CPU oracle；报告时区分 build、单元测试、数值
  accuracy、fixed-CFL stability、MPI correctness 和性能测量。正式性能报告须有同 basis
  重复测量，并同时更新 Markdown/HTML（如存在配套 HTML）。

Kunshan 默认 CPU、CPU MPI 和 root main-solver 测试依赖 GCC 16.2.0、OpenMPI 5.0.11、
CMake 4.4.3；MPI 测试设置 OMPI_MCA_coll=^hcoll。HIP/DTK 测试按文档注明的 HIP 专用
工具链例外执行。不要泄漏凭据、主机名、作业号、原始日志或未脱敏的集群路径。

每次工作结束时更新 dev 上的 living TODO，写明做了什么、测试 revision、实际工具链、
结果和剩余风险；检查 git diff --check。推送前核实目标分支和远端，不要改写已推送的
dev 历史。
```

## Current state recorded at handoff

- **Branches:** On 2026-10-09, local `master`, `origin/master`, and `upstream/master` matched. Local `dev` and `origin/dev` matched after documentation commit `27e70085`.
- **PR #203:** Open from `lql341:pr/database-type-map-init` to upstream `master`; topic commits `6f9965e0` and `1e3b2c73`. Latest checked state: mergeable, review required; deploy passed; Linux and Windows CI in progress.
- **Kunshan evidence:** `DataBaseTypeInitTest.AccessorsInitializeMappingsOnFirstUse` passed 1/1 with GCC 16.2.0, OpenMPI 5.0.11, and CMake 4.4.3 via OpenAPI.
- **Long-term solver scope:** The constrained 3D HIP vertical slice has target-node correctness and fixed-CFL stability evidence. Full multi-zone, MPI/halo, viscous/turbulence, complete state residency, and stable repeated performance claims remain unfinished.
- **Performance direction:** Investigate host boundary/ghost work (`CALC_BOUNDARY` / `NsCalcBc`) using the CPU path as oracle. Confirm the current profile and contract before implementing.

Refresh GitHub and Git state before acting; the CI and PR state above is a dated snapshot.
