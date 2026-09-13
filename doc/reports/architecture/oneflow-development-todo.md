# OneFLOW 开发待办与衔接（living document）

> 最后更新：2026-09-13
> 用途：每轮任务开始前读本文档，结束后更新本文档。让任何人或智能体
> 接手时只读这一份就能继续推进。
>
> 配套入口：[`AGENTS.md`](../../../AGENTS.md)（规则与文档地图）、
> [`ci/kunshan/README.md`](../../../ci/kunshan/README.md)（集群流程与标准套件）。

## 存储约定（长期）

本文档是 **fork-only 文档**：只存在于 lql341 的 fork 与本地仓库
（分支 `docs/development-todo`），**不进入 upstream PR，也不提交到
upstream 仓库**。原因：它是个人/团队的工作衔接记录，不属于上游项目文档。

更新流程：

```bash
git checkout docs/development-todo
# 编辑本文件（§1 状态、§2 勾选/新增、完成的移入 §5）
git commit -am "Update development todo: <一句话摘要>"
git push origin docs/development-todo
git checkout <当前工作分支>
```

基线维护：当 upstream `master` 前进（例如相关 PR 合并）后，把本分支
rebase 到新的 `upstream/master`，让它只包含本文档自身的改动，不携带
已合并的历史。

智能体配合：`oneflow-dev` 技能的 `references/workflow.md` 记录了该约定，
agent 在收尾时会走上述流程而不是 PR。

## 协作约定（重要）

### 分支模型：master + dev

- **`master`**：与 `upstream/master` 保持同步，只作为 PR 基线和只读参考。
- **`dev`**：本地与 fork 上唯一的常住工作分支。个人工作全部在这里：
  fork-only 待办文档、进行中的功能（当前含 WENO5 统一接口）、笔记等；
  推送到 `origin/dev` 即完成云端备份。
- **临时分支**：仅在需要向上游提 PR 时创建，从 `upstream/master` 分叉，
  cherry-pick `dev` 上的功能 commit（**排除文档 commit**），提完后删除。

### 不主动提 PR

- 只有收到“可以提 PR”的指示后才向 upstream 提交；在此之前所有成果留在
  `dev` 或 fork。
- fork-first 的收益：不打扰上游 CI（每次 push 到 PR 分支都会在 base 仓库
  触发 workflow 运行）、随时可 rebase、不会被半成品 PR 绑定。

### dev 的日常操作

```bash
git checkout dev
# 写代码 / 更新本文档
git add -A && git commit -m "<type>: <summary>"
git push origin dev
```

保持文档更新为**单文件、独立 commit**，方便提 PR 时用
`git cherry-pick` 精确排除。

## 本机开发环境

本机（开发工作站）有用户级 Environment Modules，优先使用而不是下载工具链：

```bash
source /home/kylinlu/Downloads/agent/repo/kylinflow/module-init.sh
module load cmake        # cmake/4.4.3
module load openmpi      # openmpi/5.0.10
```

- `module avail` 当前提供：`cmake/4.4.3`、`openmpi/5.0.10`、`kylinflow/0.1`。
- 本机角色：**推送前的快速验证**（编译 port、跑 contract test，秒级反馈）；
  目标环境验证（DTK/HIP、真实 DCU、性能数据）在昆山完成。
- 系统无 sudo、无 pip、家目录可能只读；不要在 `/tmp` 里留下需要长期保留的
  工具（tmpfs 会被清理）。

## 新会话如何开始（给智能体/接手者）

```bash
git fetch origin
git checkout dev && git pull --ff-only
cat doc/reports/architecture/oneflow-development-todo.md
```

如果工作区必须停在别的分支（例如提 PR 期间），不必切分支也能读：

```bash
git show dev:doc/reports/architecture/oneflow-development-todo.md
```

读完 §1（现状）+ §2（待办）后选任务开工；**收工前更新本文档**。本文档只存在
于 `dev` 分支（本地 + fork），master/upstream 上都没有。

## 0. 怎么用这份文档

- **开工前**：读 §1 当前状态 + §2 待办，按优先级选一件。
- **收工前**：更新 §1（分支/PR/CI 状态）、勾选或新增 §2、把完成的移到 §5。
- **交接给他人/新会话**：对方只需读本文档 + 上表中两个入口。
- 本文档是活文档，不写日期后缀；历史事实放 §5。

## 1. 当前状态快照

| 项目 | 状态 |
|---|---|
| 主分支 | `upstream/master` = PR #147 merge（含昆山 CI 脚本、TEST_PREFIX、文档） |
| 进行中的 PR | **#149**（9 commits）：路径修复 + 昆山环境要求 + 2026-09-13 性能报告 + 4-DCU 基线勘误 + 标准工作区/套件 + `AGENTS.md`/`CLAUDE.md`。CI 全绿，等上游 review |
| 分支 | `fix/contract-test-cmake-path`（#149）；`dev`（WENO5 统一接口已完成，commit `6eaf46d2` + `6f341cef` + `cc2ef93b`） |
| 昆山工作区 | 已规范化：`<workspace>/` 下 `src/`、`deps/`、`builds/`、`runs/<date>/<suite>/`、`archive/`；集群侧 README 记录具体路径 |
| 昆山作业脚本 | 四个标准套件脚本已更新到新工作区路径 |
| 智能体入口 | 仓库 `AGENTS.md` + `CLAUDE.md`；技能仓库 `oneflow-dev`（已安装到本地 skills 目录） |
| 测量口径 | 已确立：`lifecycle_*_ms` 为 repeats 总和，异口径不可比；历史 13.10× 勘误已修正为 25.55× |

**能力边界（不要越界声明）**：一维 Euler 的 CPU/HIP 后端与单节点 MPI 已实测；
CUDA、Kokkos、跨节点 MPI、完整 Navier–Stokes 主线均未验证。
主 solver 的 accelerator backend 目前只是接口骨架。

**GPU 融入路线图 (2026-09-13 启动)**：

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | FluxBackend 扩展为 Euler 多方程 Rusanov（CPU+HIP kernel） | ✅ 完成（commit `11b98029`） |
| Phase 2 | 验证桥：FluxBackend vs port EulerBackend 数值一致性 | ✅ 完成（commit `c4764c48`，机器精度一致） |
| Phase 3 | HipEulerBackend 接入 AccelBackend 统一设备管理 | ✅ 完成（commit `983641c8`） |
| Phase 4 | 主求解器 NsInvFlux 接入 FluxBackend 虚接口 | 待开始 |

## 2. 待办事项

### P0 — 等待中

- [ ] **PR #149 合并**（等上游 review；合并后同步本地 `master`，见 §4 收尾流程）。

### P1 — 下一步（建议按序）

- [x] **Phase 2：FluxBackend ↔ EulerBackend 验证桥**
  - ✅ CPU 侧 4 分辨率全部通过，误差 ≤ 2.22e-16（机器精度）。
  - 桥接测试已加入 `tests/euler/flux_backend_bridge_test.cpp`。
  - HIP 版本待昆山运行。

- [ ] **昆山 HIP contract 6/6 验证 WENO5**（WENO5 统一接口 CPU 侧已完成 8/8）
  - 在昆山用 `dcu-single` 套件跑 HIP contract test，确认 WENO5 的 GPU 路径同样通过。
  - 本地 CPU 8/8 已通过（commit `cc2ef93b`），还需昆山 HIP 6/6。

- [ ] **昆山回归 eric 的完整测试套件**
  - 范围：`tests/` 下 `task/`、`database/`、`register/`、`adt/`（框架重构的下游兼容）。
  - 方式：扩展现有 `cpu-regression` 作业脚本（增加构建/运行这些 target）。
  - 验收：全部通过；把结果补进 `dcu-single`/`cpu-regression` 的运行证据。

### P2 — 后续技术工作

- [x] **Phase 3：HipEulerBackend 接入 AccelBackend**
  - 内容：HipState 用 `AccelBackend::Allocate/Copy` 替代裸 `hipMalloc/hipMemcpy`。
  - 收益：设备选择统一由 `AccelRuntime` 管理（多 GPU 映射、环境变量），内存管理复用现有错误检查。
  - 注意：stream/event/kernel launch 保持 HIP 特定（AccelBackend 不抽象这些）。

- [ ] **Phase 4：主求解器 NsInvFlux 接入**
  - 内容：在 `NsInvFlux::Solve`/`LaxFriedrichs` 中检测 HIP backend 可用性，通过 `FluxBackend` 虚接口调用 GPU。
  - 需要上游配合或不轻易改动主求解器逻辑；可先从 `LaxFriedrichs` 单一格式开始验证。

- [ ] **GPU reduction**（优化计划阶段 D 唯一剩余项）

- [ ] **GPU reduction**（优化计划阶段 D 唯一剩余项）
  - 内容：checksum、最大误差、有限性/正状态检查放到设备端归约，只回传标量。
  - 参考：`doc/reports/architecture/oneflow-euler-optimization-plan.md` 阶段 C/D。
  - 验收：Kunshan 四规模 correctness 不变；D2H 占比进一步下降；性能复测。

- [ ] **WENO5 数值内核的 DCU 验证**（接口已稳定，CPU 8/8 通过）
  - 内容：用统一接口跑 WENO5 的 CPU/HIP 对比与四规模性能。

### P3 — 维护与清理

- [ ] 本地分支清理：`docs/kunshan-20260913-measurements`（内容已并入 #149）；
      `feat/weno5-backend-unification`（内容已合入 dev，可删除）。
- [ ] 昆山 `<workspace>/work/`、`tmp/` 定期清理（均可重建）。
- [ ] `oneflow-dev` 技能更新流程：改技能仓库 → `git push` → 各环境 `git pull`。

### 环境提醒（不属于代码任务）

- [ ] 本机家目录反复变为只读（`errors=remount-ro`），会破坏 `~/.ssh` 写入与
      SSH 控制套接字。建议排查磁盘错误并恢复可写挂载。

## 3. 接手前的关键上下文

1. **两套构建分离**：根构建是 CPU solver（MPI+METIS+CGNS）；
   一维 Euler HIP 代码在 `ports/kunshan/oneflow_1d_hip`，用 DTK 工具链独立构建。
   根构建结果不能证明 DCU 能力。
2. **四个标准套件**：`cpu-regression` / `dcu-single` / `cpu-mpi` / `dcu-mpi`；
   资源配比、通过判据、统一规模见 `ci/kunshan/README.md`。
3. **回归门槛**：数值/后端改动必须跑五算例 CPU 套件（normal `1e-8` + strict `1e-15`）；
   涉及 DCU/HIP 再加 HIP contract test。
4. **报告规范**：命名、Markdown 为源、发布边界与勘误流程见
   `doc/reports/README.md` 与 `doc/reports/performance/oneflow-euler-performance-20260913.md`。
5. **已知陷阱**：工具链（GCC 版本下限、`OMPI_SKIP_MPICXX`、CMake 依赖传递）、
   测试框架（空测试假通过、后端前缀）、节点故障——细节见 `ci/kunshan/README.md`
   和 `oneflow-dev` 技能的 `references/pitfalls.md`。

## 4. 每轮任务的收尾流程（Definition of Done）

1. 本地构建 + 相关测试套件通过；数值改动附五算例结果。
2. 推送后等待 CI 绿灯；红灯不算完成（原 #148 的教训）。
3. 更新本文档：§1 状态、§2 勾选/新增、完成的移入 §5。
4. 如产生新的测量数据：更新对应报告（Markdown + HTML），遵守命名与口径规则。
5. 如发现已发布数据的错误：修正维护中的报告并加勘误说明（参考 4-DCU 基线的处理）。

## 5. 已完成（按时间倒序）

| 日期 | 事项 | 证据 |
|---|---|---|
| 2026-09-13 | Phase 3: port HipState 接入 AccelBackend + DeviceBuffer（消除 hipMalloc/hipMemcpy 重复） | commit `983641c8` |
| 2026-09-13 | Phase 2: FluxBackend ↔ EulerBackend 数值桥接验证（4 分辨率，机器精度一致） | commit `c4764c48` |
| 2026-09-13 | Phase 1: FluxBackend 扩展为 Euler 多方程 Rusanov（CPU 验证通过：3eq 1D Euler + 5eq 3D NS）；HIP kernel 已编写待昆山验证 | commit `11b98029` |
| 2026-09-13 | WENO5 统一接口重做完成：EulerMethod 枚举、CPU/HIP Advance 分发、3 个新 contract test；根级 CMake 补链 OneDWeno5.cpp | commits `6eaf46d2`, `6f341cef`, `cc2ef93b`；本地 CPU 8/8 PASSED |
| 2026-09-13 | 昆山 CPU/DCU 全面复测：五算例 normal+strict、HIP contract 6/6、CPU/4-DCU MPI 四规模；4-DCU 与 9-02 一致（274.68 vs 277.22 ms） | `oneflow-euler-performance-20260913.md` |
| 2026-09-13 | 历史报告 4-DCU 基线勘误：13.10× → 25.55×（repeats 口径错配） | 同报告 §4.4；三份维护报告已修正 |
| 2026-09-13 | 昆山工作区规范化 + 标准四套件文档 + 作业脚本路径更新 | `ci/kunshan/README.md`；集群侧 README |
| 2026-09-13 | 修复 PR #147 遗留的 contract-test 路径（CMake + 文档链接，共 5 处） | PR #149 前两个 commit |
| 2026-09-13 | 智能体入口：`AGENTS.md`、`CLAUDE.md`、`oneflow-dev` 技能（独立公开仓库） | PR #149；github.com/lql341/oneflow-dev |
| 2026-09-13 | PR #147 合并：昆山 CI 脚本、CTest 后端前缀、初版文档 | upstream/master |
| 2026-09-02 | MPI Euler 回归与多设备 HIP 验证（4-rank/4-DCU） | commit `6ae9fa41` |
