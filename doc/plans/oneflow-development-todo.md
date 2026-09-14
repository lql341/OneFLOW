# OneFLOW 开发待办与衔接（living document）

> 最后更新：2026-09-14（本轮：E1 与 E2.1–E2.3 完成；E2.4 INIT/restart 证据待做）
> 用途：每轮任务开始前读本文档，结束后更新本文档。让任何人或智能体
> 接手时只读这一份就能继续推进。
>
> 配套入口：[`AGENTS.md`](../../../AGENTS.md)（规则与文档地图）、
> [`ci/kunshan/README.md`](../../../ci/kunshan/README.md)（集群流程与标准套件）。

## 存储约定（长期）

本文档是 **fork-only 文档**：只存在于 lql341 的 fork 与本地仓库
（分支 `dev`），**不进入 upstream PR，也不提交到 upstream 仓库**。
原因：它是个人/团队的工作衔接记录，不属于上游项目文档。

更新流程：

```bash
# 本文档在 dev 分支上，直接编辑即可
vim doc/plans/oneflow-development-todo.md
git commit -am "Update development todo: <一句话摘要>"
git push origin dev
```

基线维护：当 upstream `master` 前进（例如相关 PR 合并）后，把 `dev`
rebase 到新的 `upstream/master`，让它只包含本文档 + 开发中的功能，
不携带已合并的历史。提 PR 时从 `upstream/master` 分临时分支 cherry-pick
功能 commit（排除文档 commit）。

智能体配合：`oneflow-dev` 技能的 `references/workflow.md` 记录了该约定。

## 协作约定（重要）

### 分支模型：master + dev

- **`master`**：与 `upstream/master` 保持同步，只作为 PR 基线和只读参考。
- **`dev`**：本地与 fork 上唯一的常住工作分支。个人工作全部在这里：
  本文档、进行中的功能、笔记等；推送到 `origin/dev` 即完成云端备份。
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
module load cmake/4.2.0
module load openmpi/4.1.4
module load oneflow/metis-5.1.0
module load oneflow/cgns-3.4.0
```

- `module avail` 当前机器已验证：`cmake/4.2.0`、`openmpi/4.1.4`、`oneflow/metis-5.1.0`、`oneflow/cgns-3.4.0`；目标节点以 `ci/kunshan/README.md` 为准。
- 本机角色：**推送前的快速验证**（编译 port、跑 contract test，秒级反馈）；
  目标环境验证（DTK/HIP、真实 DCU、性能数据）在昆山完成。
- 系统无 sudo、无 pip、家目录可能只读；不要在 `/tmp` 里留下需要长期保留的
  工具（tmpfs 会被清理）。

## 新会话如何开始（给智能体/接手者）

```bash
git fetch origin
git checkout dev && git pull --ff-only
cat doc/plans/oneflow-development-todo.md
```

如果工作区必须停在别的分支（例如提 PR 期间），不必切分支也能读：

```bash
git show dev:doc/plans/oneflow-development-todo.md
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
| 主分支 | `master` = `origin/master` = `upstream/master` = `90749492`（已同步） |
| 进行中的 PR | 无；功能继续留在 fork `dev`，未授权不主动提 PR |
| 分支 | 本地 `dev` = 最新上游基线 + accelerator Phase 1-3 + 架构 contract WIP；`origin/dev` 尚未强推更新 |
| 昆山工作区 | 已规范化：`<workspace>/` 下 `src/`、`deps/`、`builds/`、`runs/<date>/<suite>/`、`archive/`；集群侧 README 记录具体路径 |
| 昆山作业脚本 | 四个标准套件脚本已更新到新工作区路径 |
| 智能体入口 | 仓库 `AGENTS.md` + `CLAUDE.md`；技能仓库 `oneflow-dev`（已安装到本地 skills 目录） |
| 测量口径 | 已确立：`lifecycle_*_ms` 为 repeats 总和，异口径不可比；历史 13.10× 勘误已修正为 25.55× |

**能力边界（不要越界声明）**：一维 Euler 的 CPU/HIP 后端与单节点 MPI 已实测；
CUDA、Kokkos、跨节点 MPI、完整 Navier–Stokes 主线均未验证。
`codes/accel` 的 accelerator substrate 已完成 Phase 1-3；主 solver 的 NS/Euler accelerator execution path 仍未接入。

**GPU 融入路线图 (2026-09-13 启动)**：

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | FluxBackend 扩展为 Euler 多方程 Rusanov（CPU+HIP kernel） | ✅ `11b98029` |
| Phase 2 | 验证桥：FluxBackend vs port EulerBackend 数值一致性 | ✅ `c4764c48`（机器精度 2.22e-16） |
| Phase 3 | HipEulerBackend 接入 AccelBackend + DeviceBuffer 统一设备管理 | ✅ `983641c8` |
| Phase 4 | 主求解器 UNsInvFlux::CalcInvFlux 批量 GPU 化 | 🔜 下一步 |

**融合后的唯一执行主线**：架构 contract 解决生命周期/所有权，FluxBackend 解决批量通量计算；两者在主 solver CPU adapter 汇合，再复用到 HIP/DCU。

| 融合层 | 交付物 | 状态 |
|---|---|---|
| A. 运行基线 | master 与 origin/upstream 同步 | ✅ |
| B. standalone | 1D Euler CPU/HIP lifecycle、FullTrace/NoTrace、MPI 实验 | ✅ |
| C. accel substrate | AccelRuntime、AccelBackend、FluxBackend、DeviceBuffer | ✅ Phase 1-3 |
| D. domain contract | EulerDomain views、StateRegistry；通用 views 的布局/几何/能力元数据已补齐 | 🟨 E1 完成；E2 owner 已接入，生产钩子待做 |
| E. CPU vertical slice | INIT_FLOWFIELD、CPU adapter、RungeKutta、CPU oracle | ⬜ 当前主任务 |
| F. DCU vertical slice | 同一 adapter 切换 HIP/DCU，保留 capability guard 和 fallback | ⬜ |
| G. MPI/性能 | host-staged halo、GPU-aware probe、reduction、性能 | ⬜ |

**整合约束**：`FluxBackend` 当前接收 equation-major conserved face state；`UNsInvFlux` 当前提供 reconstructed primitive state，且 area 由旧路径单独处理。接入前必须完成变量转换、面积/法向约定和 face connectivity 对齐，并用旧 CPU 逐面路径做 oracle。

**port / accel / NS 三层关系（统一后）**：

| 层 | 角色 | GPU 状态 |
|----|------|----------|
| **port** (`ports/kunshan/`) | 最小实验闭环：1D Euler, persistent state, FullTrace, 合约测试 | ✅ 完整 |
| **accel** (`codes/accel/`) | 生产基础设施：AccelRuntime, FluxBackend, DeviceBuffer | ✅ 完整（标量+Euler） |
| **NS** (`codes/ns/` + `codes/uns/`) | 主求解器：5 方程, 8 种通量格式, 逐面 CPU 循环 | ❌ 待接入 |

**OpenAI NS 解决公告 (2026-09-08) 参考**：
- 数学证明（奇点存在性），非数值求解器，与 OneFLOW 直接技术关联有限
- 方法论可借鉴：Euler 先做热身 → NS；形式化验证 1/6 工时（对应 Phase 2 桥接测试）
- 行业信号：AI+CFD 交叉领域在加速，GPU 求解器基础设施具有战略价值

## 2. 待办事项

### P0 — dev 融合基线

- [x] 最新 `origin/master` / `upstream/master` 已同步到本地 `master`。
- [x] 协同开发 Phase 1-3、旧架构 contract/StateRegistry 已统一恢复到本地 `dev`。
- [x] standalone CPU、根工程、根 CTest、CPU bridge 已完成本机验证。
- [ ] 完成融合后 dev 的 contract/adapter 验证，再更新 `origin/dev`。

### P1 — 主 solver CPU vertical slice（按序）

- [ ] **大阶段 E：CPU vertical slice**
  - [x] E1：完成 domain contract 泛化，保留 1D Euler specialization。
    - [x] E1.1：明确 field representation、equation layout、face-area ownership。
    - [x] E1.2：补充 cell/face geometry view 和 face connectivity view。
    - [x] E1.3：补充 ghost/halo 元数据与 3/5 方程 capability 声明。
    - [x] E1.4：为 view shape、layout 和 geometry 约束补 contract tests。
  - [ ] E2：完成 StateRegistry 生产化。
    - [x] E2.1：确定 `SimuImp`/`FieldSimu` 生命周期 owner，不把 state 塞入 kernel。
    - [x] E2.2：覆盖 solver + zone + grid level + backend/device identity。
    - [x] E2.3：接入 create/reuse/invalidate/clear 生命周期钩子。
    - [ ] E2.4：覆盖重复创建、缺失 state、restart invalidate 的测试。
  - [ ] E3：完成 CPU adapter。
    - [ ] E3.1：定义 primitive-to-conserved 的 3/5 方程转换。
    - [ ] E3.2：按 equation-major 约定 pack face state。
    - [ ] E3.3：建立 `FaceStateView` → `CpuFluxBackend` 调用。
    - [ ] E3.4：按显式 connectivity map 回写 residual。
    - [ ] E3.5：保留旧逐面路径作为 feature-gated fallback。
  - [ ] E4：接入 INIT_FLOWFIELD/restart。
    - [ ] E4.1：初始化完成后 create + Upload。
    - [ ] E4.2：restart 后 invalidate + Upload，禁止复用旧 device state。
    - [ ] E4.3：补初始化/重启状态一致性测试。
  - [ ] E5：接入 RungeKutta。
    - [ ] E5.1：增加 solver-aware fast path capability check。
    - [ ] E5.2：fast path 与现有 task 序列保持同一 stage 顺序。
    - [ ] E5.3：不满足能力时回退原 task 序列。
  - [ ] E6：建立 CPU oracle 与主 solver 验收门。
    - [ ] E6.1：新 batch 路径 vs 旧逐面路径逐 face/逐 equation 对照。
    - [ ] E6.2：检查 finite、positive density/pressure、conservation。
    - [ ] E6.3：补 lifecycle、state reuse、invalid request、batch equality CTest。
    - [ ] E6.4：完成融合后 dev 的 contract/adapter 验证，再更新 `origin/dev`。

### P1 — DCU 与回归验证

- [ ] 昆山 HIP contract 6/6 验证 WENO5。
- [ ] 主 solver CPU batch 通过后，把同一 adapter 切换到 HIP/DCU，并在昆山做 correctness。
- [ ] 昆山回归 eric 的完整 `task/database/register/adt` 测试套件。

### P2 — 后续技术工作

- [ ] **GPU reduction**（优化计划阶段 D 唯一剩余项）
  - 内容：checksum、最大误差、有限性/正状态检查放到设备端归约，只回传标量。
  - 参考：`doc/plans/oneflow-euler-optimization-plan.md` 阶段 C/D。
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
| 2026-09-14 | E2 registry lifecycle：`GetOrCreate`/`Invalidate`/`Clear` 已接入 registry，覆盖 create/reuse/invalidate/teardown 语义；restart task hook 仍待接入 | `codes/accel/include/EulerDomainStateRegistry.h`; `codes/accel/src/EulerDomainStateRegistry.cpp`; registry 4/4；SimuContext 12/12 |
| 2026-09-14 | E2 registry owner：`SimuContext` 持有 accelerator state registry，teardown 先清 state 再 finalize runtime；key contract 覆盖 solver/zone/grid/backend | `codes/main/include/SimuContext.h`; `codes/main/src/SimuContextEnv.cpp`; `tests/main/simu_context_test.cpp`; 12/12；根 target 编译阶段通过（最终链接受构建目录 METIS cache 影响） |
| 2026-09-14 | E1 domain contract 元数据：补充 field layout/representation、face-area policy、geometry/connectivity、ghost/halo 与 3/5 方程 capability，并新增 5 项 contract assertions | `codes/accel/include/AccelViews.h`; `tests/euler_domain_contract_test.cpp`; contract test 5/5；bridge 4/4；根工程增量编译通过 |
| 2026-09-14 | 融合两条路线：最新 upstream 基线 + accelerator Phase 1-3 + EulerDomain/StateRegistry contract，统一进入本地 `dev` | `dev`；原 WIP 已从 stash 恢复并提交 |
| 2026-09-14 | 融合后本机验证：根工程编译 100%，根 CTest 163/163 通过，domain contract 3/3、StateRegistry 3/3 | `/tmp/oneflow-merged-dev-root`；1 个测试明确 Disabled |
| 2026-09-14 | 将架构 contract 测试注册到根 CTest 路由 | commit `6fc67556` |
| 2026-09-14 | 修复 Phase 3 引入的 standalone CPU runner 依赖：AccelRuntime、CpuBackend、OneDWeno5.cpp | commit `b07dd294` |
| 2026-09-14 | CPU runner、FluxBackend bridge 与根工程验证闭环 | bridge 最大差异 `2.22e-16`；root build 100% |
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
