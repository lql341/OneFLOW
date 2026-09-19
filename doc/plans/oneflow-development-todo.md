# OneFLOW 开发待办与衔接（living document）

> 最后更新：2026-09-16（本轮收口：3D m6 公平 8 CPU ranks vs 1 DCU timing 与 MPI rank-local state-sync 修复完成；50-step 稳定性和 fresh CPU 五 case 门禁仍是 blocker）
> 用途：每轮任务开始前读本文档，结束后更新本文档。让任何人或智能体
> 接手时只读这一份就能继续推进。
>
> 配套入口：[`AGENTS.md`](../../../AGENTS.md)（规则与文档地图）、
> [`ci/kunshan/README.md`](../../../ci/kunshan/README.md)（集群流程与标准套件）。

## 本轮新增证据（2026-09-16）

- 标准 runner：`ci/kunshan/f3-main-solver-benchmark.slurm`；3D stage runner 使用流式 verifier，避免多步 trace 一次性读入造成 OOM。
- 精度门禁：3-step、3-stage RK、36 条 trace 全通过；HIP 对 legacy 最大绝对差 `2.8399504969911504e-13`，最大 scaled error `1.992850329202156e-13`。
- 性能 basis：同一输入、`steps=3`、warmup `1`、repeats `3`，端到端 wall-clock；legacy `24775.415 ms`，CPU batch `25520.782 ms`（`0.970794x`），HIP/DCU batch `25224.514 ms`（`0.982196x`）。结论是当前 host-staged 单卡路径准确但未加速。
- 公平资源口径：legacy/CPU batch 使用 8 CPU MPI ranks，HIP batch 使用 1 DCU；同一 basis 下分别为 `25904.228096 ms`、`26794.339157 ms`（`0.966780x`）、`25658.172501 ms`（`1.009590x`）。3-step trace 36 条通过；当前仅约 `0.95%` 优势，不作为有意义加速结论。
- MPI 修复：`SyncAllEulerDomainStates` 改为按 `ZoneState::localZid` 遍历 rank-local zones，避免非 owner rank 解引用空全局 grid；修复后 8-rank CPU warmup、trace 与 HIP contract 通过。
- 回归边界：标准隔离 CPU runner 的 normal/strict fresh 重跑在 continuation fixture 上因“solver 返回 0 但目标结果文件不存在”停止；当前及上一版 restart fixture 均复现，需单独修复 fixture/runner 后才能把本轮修改标为 fresh CPU regression green。
- 未闭环项保持不变：50-step CPU/HIP 共同物理发散，MPI/多卡、完整 NS/HIP 与 GPU-resident 性能仍未完成。

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

基线维护：当 upstream `master` 前进（例如相关 PR 合并）后，把
`upstream/master` **merge** 进 `dev`（`git merge upstream/master`）。
`dev` 已推送到 `origin/dev` 且是日常分支，**不要用 rebase**：那会改写已公开的
提交历史，与「推送到 `origin/dev` 即完成云端备份」矛盾。既有的 `6d30b783`、
`e584bcbb` 都是 merge commit。只有确认改动尚未推送到 `origin/dev` 时才考虑
rebase。提 PR 时从 `upstream/master` 分临时分支 cherry-pick
功能 commit（排除文档 commit）。

智能体配合：`oneflow-dev` 技能的 `references/workflow.md` 记录了该约定。

## 协作约定（重要）

### 分支模型：master + dev

- **`master`**：与 `upstream/master` 保持同步，只作为 PR 基线和只读参考。
- **`dev`**：本地与 fork 上唯一的常住工作分支。个人工作全部在这里：
  本文档、进行中的功能、笔记等；推送到 `origin/dev` 即完成云端备份。
- **临时分支**：仅在需要向上游提 PR 时创建，从 `upstream/master` 分叉，
  cherry-pick `dev` 上的功能 commit（**排除文档 commit**），提完后删除。

### 提 PR 时的两个坑

1. **`AGENTS.md` 不能整体覆盖。** dev 上的文档地图指向
   `doc/handoff/oneflow-project-handoff-20260903.md`，upstream 上指向
   `doc/reports/architecture/oneflow-project-handoff-20260903.md`——两边各自正确
   （dev 已删除 `doc/reports/architecture/`，upstream 仍保留）。所以 `AGENTS.md`
   只能**增量 cherry-pick**；整体覆盖会把上游那条链接改坏，且 cherry-pick 时
   会因上下文含该行而冲突。
2. **cherry-pick 按 SHA，不要按标题。** dev 历史里同一改动存在两份（一份来自
   dev 自身，一份随 `origin/master` 合入），例如 Phase 3、FluxBackend 多方程
   Rusanov、FluxBackend↔EulerBackend 桥接测试、WENO5/EulerMethod 都有重名提交。
   按标题挑会挑到错的那个。

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
| 主分支 | `master` = `origin/master` = `upstream/master` = `fa3f3b06`（三端 0/0；已合入 dev） |
| 进行中的 PR | 无；功能继续留在 fork `dev`，未授权不主动提 PR |
| 分支 | 本地 `dev` = `origin/dev` = `e584bcbb`；`f2b3d43c` 已推送，且已把 `upstream/master` `fa3f3b06` 合入 `dev`（dev 相对 upstream ahead 96 / behind 0）；F2.4 3D m6 1-step/3-step 精度门禁与公平 timing 已补齐，50-step 和 fresh CPU 五 case 门禁仍是 blocker |
| 昆山工作区 | 已规范化：`<workspace>/` 下 `src/`、`deps/`、`builds/`、`runs/<date>/<suite>/`、`archive/`；集群侧 README 记录具体路径 |
| 昆山作业脚本 | 四个标准套件脚本已更新到新工作区路径 |
| 智能体入口 | 仓库 `AGENTS.md`（含文档地图、分支模型与工作规则）；`CLAUDE.md` 已于 2026-09-19 删除；技能仓库 `oneflow-dev`（已安装到本地 skills 目录） |
| 测量口径 | 已确立：`lifecycle_*_ms` 为 repeats 总和，异口径不可比；历史 13.10× 勘误已修正为 25.55× |
| 当前进度 | E1–E6、F1、F2.1–F2.4 的 1-step/RK 与 3-step 门禁已完成；标准 root runner 与单卡 3-step benchmark 已完成，但 50-step 在 CPU/HIP 共用配置下共同发散，当前未观察到端到端 DCU 加速。 |
| 最新验证 | `122246208`：昆山 3D m6、896256 faces、3-stage RK、3 steps，36 条三路 trace 记录通过；HIP 对 legacy 最大绝对差 `2.8399504969911504e-13`，scheduler/workload `COMPLETED/0:0`。同一 `steps=3, warmup=1, repeats=3` basis 下，legacy `24775.415 ms`，CPU batch `25520.782 ms`，HIP/DCU batch `25224.514 ms`，加速比 `0.982196x`；50-step 仍共同出现负压/Inf/NaN。 |

**能力边界（不要越界声明）**：一维 Euler 的 CPU/HIP 后端与单节点 MPI 已实测；
CUDA、Kokkos、跨节点 MPI、完整 Navier–Stokes 主线均未验证。
`codes/accel` 的 accelerator substrate 已完成 Phase 1-3；主 solver 已有受控的 CPU/HIP inviscid batch seam，并已完成 root HIP 编译、contract、adapter one-call 与小 case 1-step。真实主 solver HIP 3D m6 三阶段 1-step 已验证；50-step 稳定性、标准 root runner、完整 NS/Euler accelerator path 仍未验证。

**GPU 融入路线图 (2026-09-13 启动)**：

| 阶段 | 内容 | 阶段含义（中文） | 状态 |
|------|------|------------------|------|
| Phase 1 | FluxBackend 扩展为 Euler 多方程 Rusanov（CPU+HIP kernel） | 先让批量通量 backend 能处理标量、3 方程 Euler 和 5 方程 NS 数据。 | ✅ `11b98029` |
| Phase 2 | 验证桥：FluxBackend vs port EulerBackend 数值一致性 | 用独立桥接测试证明新 backend 与已有 CPU oracle 的数值结果一致。 | ✅ `c4764c48`（机器精度 2.22e-16） |
| Phase 3 | HipEulerBackend 接入 AccelBackend + DeviceBuffer 统一设备管理 | 统一 accelerator runtime 和设备内存管理，避免 backend 各自维护重复资源。 | ✅ `983641c8` |
| Phase 4 | 主求解器 UNsInvFlux::CalcInvFlux 批量 GPU 化 | 把生产主 solver 的逐面通量循环改造成可切换的批量 CPU/HIP/DCU 路径。 | 🟨 3D m6 3-stage 1-step 已通过；50-step 稳定性与标准 runner 待收口 |

**融合后的唯一执行主线**：架构 contract 解决生命周期/所有权，FluxBackend 解决批量通量计算；两者在主 solver CPU adapter 汇合，再复用到 HIP/DCU。

| 融合层 | 交付物 | 阶段含义（中文） | 状态 |
|---|---|---|---|
| A. 运行基线 | master 与 origin/upstream 同步 | 先固定共同代码基线，确保两条开发线从同一个上游版本继续。 | ✅ |
| B. standalone | 1D Euler CPU/HIP lifecycle、FullTrace/NoTrace、MPI 实验 | 用最小可控算例验证状态生命周期、CPU/HIP 后端和 MPI 基础能力。 | ✅ |
| C. accel substrate | AccelRuntime、AccelBackend、FluxBackend、DeviceBuffer | 建立与具体 solver 解耦的运行时、设备内存和批量 kernel 基础设施。 | ✅ Phase 1-3 |
| D. domain contract | EulerDomain views、StateRegistry；通用 views 的布局/几何/能力元数据已补齐 | 明确 solver 与 accelerator 之间的数据、所有权和生命周期契约。 | ✅ E1、E2 与生命周期 service、主 solver INIT/restart hook 已完成 |
| E. CPU vertical slice | INIT_FLOWFIELD、CPU adapter、RungeKutta、CPU oracle | CPU 主 solver 初始化、通量、时间推进、逐面 trace 与物理门禁已闭环。 | ✅ E1–E6 |
| F. DCU vertical slice | root 生产 HIP 编译、smoke、contract、adapter one-call、小 case 1-step/物理语义、3D m6 1-step 与同 revision CPU 回归已闭环；50-step 稳定性、标准 runner、完整 NS/MPI/性能尚未完成 | F2.4 3-stage 1-step 已证明 3D inviscid seam；50-step 共同发散，不能归因于 HIP 分歧。 | 🟨 |
| G. MPI/性能 | host-staged halo、GPU-aware probe、reduction、性能 | 最后处理跨 rank 数据交换、设备归约和端到端规模化性能。 | ⬜ |

**整合约束：** `FluxBackend` 接收 equation-major conserved face state；`UNsInvFlux` 提供 reconstructed primitive state，adapter 负责转换，backend 负责面面积；face connectivity 仍由主 solver 的 `AddF2CField` 处理。CPU batch 入口必须以旧 CPU 逐面路径为 oracle。

**port / accel / NS 三层关系（统一后）**：

| 层 | 角色 | GPU 状态 |
|----|------|----------|
| **port** (`ports/kunshan/`) | 最小实验闭环：1D Euler, persistent state, FullTrace, 合约测试 | ✅ 完整 |
| **accel** (`codes/accel/`) | 生产基础设施：AccelRuntime, FluxBackend, DeviceBuffer | ✅ 完整（标量+Euler） |
| **NS** (`codes/ns/` + `codes/uns/`) | 主求解器：5 方程；当前仅 Lax-Friedrichs HIP batch opt-in 满足 F1 capability | 🟨 已接 seam/guard，待真实主 solver 数值执行 |

**OpenAI NS 解决公告 (2026-09-08) 参考**：
- 数学证明（奇点存在性），非数值求解器，与 OneFLOW 直接技术关联有限
- 方法论可借鉴：Euler 先做热身 → NS；形式化验证 1/6 工时（对应 Phase 2 桥接测试）
- 行业信号：AI+CFD 交叉领域在加速，GPU 求解器基础设施具有战略价值

## 2. 待办事项

### P0 — dev 融合基线

- [x] 最新 `origin/master` / `upstream/master` 已同步到本地 `master`。
- [x] 协同开发 Phase 1-3、旧架构 contract/StateRegistry 已统一恢复到本地 `dev`。
- [x] standalone CPU、根工程、根 CTest、CPU bridge 已完成本机验证。
- [x] 完成融合后 dev 的 contract/adapter 验证并更新 `origin/dev`。
- [x] 已将 `upstream/master` `a6c81105` 合并到 dev；保留 mutable `SimuContext&` 与 StateRegistry 生命周期，并吸收 upstream 的分阶段 `FieldSimu`、policy 和 parser 测试。

### P1 — 主 solver CPU vertical slice（按序）

- [x] **大阶段 E：CPU vertical slice** —— 在 CPU 主 solver 上完成第一条可验证的端到端加速纵切线。
  - [x] E1：完成 domain contract 泛化，保留 1D Euler specialization。说明：定义数据布局、几何、连通性和能力边界，但不改变旧 solver 行为。
    - [x] E1.1：明确 field representation、equation layout、face-area ownership。
    - [x] E1.2：补充 cell/face geometry view 和 face connectivity view。
    - [x] E1.3：补充 ghost/halo 元数据与 3/5 方程 capability 声明。
    - [x] E1.4：为 view shape、layout 和 geometry 约束补 contract tests。
  - [x] E2：完成 StateRegistry 生产化。说明：让每次 solver/zone/grid/backend 执行拥有可复用、可失效且按顺序释放的 state。
    - [x] E2.1：确定 `SimuImp`/`FieldSimu` 生命周期 owner，不把 state 塞入 kernel。
    - [x] E2.2：覆盖 solver + zone + grid level + backend/device identity。
    - [x] E2.3：接入 create/reuse/invalidate/clear 生命周期钩子。
    - [x] E2.4：覆盖重复创建、缺失 state、restart invalidate 的测试。
  - [x] E3：完成 CPU adapter。说明：把主 solver 的 primitive face 数据转换、批量通量计算和 residual 回写串成一条 CPU 路径。
    - [x] E3.1：定义 primitive-to-conserved 的 3/5 方程转换。
    - [x] E3.2：按 equation-major 约定 pack face state。
    - [x] E3.3：建立 `FaceStateView` → `CpuFluxBackend` 调用。
    - [x] E3.4：按显式 connectivity map 回写 residual。
    - [x] E3.5：主 solver `UNsInvFlux` 增加 `ONEFLOW_ENABLE_UNS_CPU_BATCH=1` 的 3D 五方程 CPU batch 入口；仅在 CPU + Lax-Friedrichs 时启用，其他情况保留旧逐面 fallback。
  - [x] E4：接入 INIT_FLOWFIELD/restart。说明：初始化和重启都必须显式建立或刷新 backend state，不能复用过期设备数据。
    - [x] E4.1：新增生命周期 service，保证初始化先 invalidate，再 create + Upload，成功后才进入 registry。
    - [x] E4.2：restart 显式 invalidate 后重新 create + Upload，失败时不登记半初始化 state。
    - [x] E4.3：补 mock backend 测试，覆盖初始化替换、restart fresh state、Upload 失败不登记，以及 3/5 方程 field contract。
    - [x] E4.4：把生命周期 service 接入实际 `INIT_FLOWFIELD`、`READ_RESTART` task chain，并绑定生产 solver 的 MRField view。
      - [x] E4.4a：实现生产 CPU `EulerDomainBackend`，完成 equation-major host Upload/Download、CPU key 校验和 Advance capability guard。q 的 internal/boundary/ghost extent 仍由 E4.4c 的 MRField adapter 明确。
      - [x] E4.4b：`ISimuTask::Execute` 与 `SolveFieldTask`/`FieldSimu` 已改为显式接收同一个可变 `SimuContext` owner；未引入隐式全局 registry。
      - [x] E4.4c：在 `INIT_FIRST`、`INIT_RESTART`/`READ_RESTART` 完成后按 solver/zone/grid key 调用 Initialize/Restart hook。
        - [x] E4.4c.1：`MRField` internal-cell → contiguous equation-major snapshot adapter，boundary-cell 数量保留为 `nGhostCells` metadata。
        - [x] E4.4c.2：`FieldSimu` 在 `INIT_FLOWFIELD` 完成后遍历 solver/grid，按 `startStrategy` 调用 Initialize 或 Restart；production object compile 已通过。
        - [x] E4.4c.3：真实 `m6wingroe_sa` root case 初始化 50 步通过，restart 副本 `startStrategy=1` 正常返回；初始化输出 residual baseline 通过。
  - [x] E5：接入 RungeKutta。说明：把 RK 的宏步/阶段调度接入主 solver 的 `TimeIntegral`，在能力不满足时透明回退旧 task 序列；这是 3D 主 solver 的 CPU 调度纵切线，不等于 standalone 1D 数值内核或 DCU kernel。
    - E5 前置边界（2026-09-14）：当前 `EulerDomainState` 仍是 internal-cell lifecycle snapshot，主 solver 的 MRField/task path 保持数值权威；E5 只把 backend `Advance` 用作显式 host stage scheduler，face geometry/connectivity、halo、primitive/conserved kernel ownership 留给 E6/F。
    - [x] E5.1：增加 solver-aware fast path capability check。说明：由 `TimeIntegral` 根据 solver/zone/grid/方程/通量/时间积分/物理模型条件决定是否允许 CPU RK seam。
      - [x] E5.1a：新增独立 capability contract，明确 NS/单 local zone/finest grid/5 方程/Lax/显式 RK/无 viscous-source-limiter-interface 条件，并返回具体 fallback 原因；已接入 context-aware `TimeIntegral`。
    - [x] E5.2：fast path 与现有 task 序列保持同一 stage 顺序。说明：`LOAD_Q`、`CALC_TIME_STEP` 先执行，随后每个 stage 严格执行 `LOAD_RESIDUALS → UPDATE_RESIDUALS → CALC_LHS → UPDATE_FLOWFIELD → CALC_BOUNDARY`。
    - [x] E5.3：不满足能力时回退原 task 序列。说明：guard reject、缺少 registry state 或空 RK coefficient 时调用原 `RungeKutta()`，既有 viscous 3D case 已完成 1-step runtime 验证。
  - [x] E6：建立 CPU oracle 与主 solver 验收门。说明：只有逐面数值一致、物理量有效且回归通过，CPU vertical slice 才能算完成。
    - [x] E6.1：完成 3D m6wing Lax-Friedrichs 主 solver 的 legacy vs batch 50-step 输出级 oracle；`aero/res/turbres` 完全一致，`wallaero` 最大绝对差约 1.00e-12、最大相对差约 2.79e-11。
      - [x] E6.1a：新增 opt-in ONEFLOW_UNS_TRACE_FILE 二进制 trace，比较 qf1/qf2/invflux 的逐 face/逐 equation 数值；896256 faces × 5 equations，invflux 最大绝对差 1.11e-15、最大相对差 1.41e-10。
    - [x] E6.2：检查 finite、positive density/pressure、conservation。trace 两侧均 finite，最小 density 约 0.999999999998、最小 pressure 约 1.0135158932；adapter 内部面守恒 CTest 通过。
    - [x] E6.3：补 lifecycle、state reuse、invalid request、batch equality CTest；全量 CTest 187/187 通过，另有 1 个既有测试 Disabled。
    - [x] E6.4：完成融合后 dev 的 contract/adapter 验证，并按确认更新 `origin/dev`。
      - [x] E6.4a：在昆山 CPU 队列完成融合后 dev 的 contract/adapter 与主 solver 回归验证；origin/dev 推送另行处理。
      - [x] E6.4b：已将 merge 后 dev 推送到 `origin/dev`（`6d30b783`）。

    **下一阶段执行顺序：** E6.4b 已完成；进入 F 阶段，把 CPU 已验证的 adapter seam 切换到 HIP/DCU，并重新执行同一组 trace/物理不变量门禁。

### P1 — DCU 与回归验证

说明：CPU vertical slice 通过后，才在昆山真实 DCU 节点验证 HIP/DCU 和完整回归套件。


- [x] 昆山 standalone 1D HIP contract：DTK 26.04、`gfx906`、`dcu:1`，GoogleTest/CTest 均 9/9 通过，含 WENO5 与 CPU oracle 对照。
- [x] HIP CTest 统一架构：standalone 与根工程共用 `cmake/OneFLOWEulerContract.cmake`；根工程默认关闭，`ONEFLOW_ENABLE_HIP_TESTS=ON` 才注册硬件测试；Kunshan runner 用 `hardware` label + `HIP` 前缀门控，避免无 DCU 时阻塞普通 CPU CTest。
- [ ] **F：主 solver DCU vertical slice**。说明：把已通过 CPU oracle 的主 solver batch seam 切换到 HIP/DCU；standalone HIP 通过不等于主 solver DCU 已完成。
  - [x] F1：主 solver HIP backend registration。说明：在不改变 CPU 默认路径的前提下，将 HIP backend 与 capability/fail-fast guard 接入生产 solver；`HipFluxBackend` 为无状态执行对象，不冒充 `EulerDomainState` 生命周期。
    - [x] F1.1：`d5005ad6` 完成统一 `FluxBackend&` 注入与 CPU/HIP 共用 pack；`ONEFLOW_ENABLE_HIP_TESTS=ON` 联动生产 HIP backend。
    - [x] F1.2：`ca14118c` 将 `UNsInvFlux` HIP path 限定为受支持 solver、5 方程、Lax-Friedrichs、单 local zone、finest/single grid，并逐项返回稳定拒绝原因。
    - [x] F1.3：默认/CPU 路径不变；显式请求 HIP 而 build/runtime/backend/capability 不满足时 fail-fast，不静默回退。
  - [ ] F2：主 solver CPU/HIP numerical gate。说明：先用小 case 复现 CPU oracle，再逐步扩大到 3D m6 case。
    - [x] F2.1：adapter one-call CPU/HIP oracle；257 faces × 5 equations，覆盖 3D normals、ALE mesh-normal velocity、face area 与非 boundary-first 显式 mask；flux 最大差 `6.661e-16`，residual 最大差 `1.110e-15`。
    - [x] F2.2：HIP 小 case 1-step LU-SGS 与 legacy CPU、CPU batch 对比，检查 `qf1`、`qf2`、face flux、residual、state trace。
    - [x] F2.3：小 case 检查 finite、positive density/pressure、boundary semantics、conservation、ALE 与 face-area ownership。
    - [ ] F2.4：扩大到 3D m6 case，并完成长步稳定性门禁。
      - [x] F2.4a：3D m6 896256 faces、Lax-Friedrichs、3-stage RK 1-step；HIP 对 legacy 最大差 `1.706e-13`，finite 且 density/pressure 为正。
      - [x] F2.4a-3：3-step 三路 trace 共 36 条记录通过；HIP 对 legacy 最大绝对差 `2.8399504969911504e-13`，最大 scaled error `1.992850329202156e-13`。
      - [ ] F2.4b：50-step 稳定性；CPU/HIP 共用配置约第 20 步共同出现负压/Inf/NaN，需先修复 CFL/边界/初始化/物理稳定性条件。
  - [ ] F3：主 solver DCU target-node evidence。说明：记录 DTK、gfx906、visible device、资源 tuple 和 workload exit code。
    - [x] F3.1：标准 root HIP runner 已沉淀；支持 trace step 参数化、精度门禁后 benchmark、同 basis timing 与非空 HIP test 检查。
    - [x] F3.2：既有 CPU queue regression evidence 与 DCU 构建/contract/adapter one-call/小 case/3-step trace/标准 runner/单卡 timing 已完成；50-step、MPI/多卡与 GPU-resident 性能不提前标记。
    - [ ] F3.3：修复 continuation fixture/runner 的 fresh output 生成问题，再重新执行本轮修改的 CPU normal `1e-8` + strict `1e-15` 五 case 门禁。
- [ ] 昆山回归 eric 的完整 `task/database/register/adt` 测试套件。

### P2 — 后续技术工作

说明：功能正确性稳定后，再做设备归约、WENO5 DCU 验证和性能优化。


- [ ] **GPU reduction**（优化计划阶段 D 唯一剩余项）
  - 内容：checksum、最大误差、有限性/正状态检查放到设备端归约，只回传标量。
  - 参考：`doc/plans/oneflow-euler-optimization-plan.md` 阶段 C/D。
  - 验收：Kunshan 四规模 correctness 不变；D2H 占比进一步下降；性能复测。

- [ ] **WENO5 数值内核的 DCU 四规模验证**（单卡 contract 已完成）
  - [x] `nx=32` contract：WENO5 HIP 与 CPU oracle 对齐。
  - [ ] `nx=65536/262144/1048576/4194304` correctness 与性能复测。

### P3 — 维护与清理

说明：清理临时分支和工作区，并把验证后的协作流程沉淀回技能与文档。


- [ ] 本地分支清理：`docs/kunshan-20260913-measurements`（内容已并入 #149）；
      `feat/weno5-backend-unification`（内容已合入 dev，可删除）。
- [ ] 昆山 `<workspace>/work/`、`tmp/` 定期清理（均可重建）。
- [ ] `oneflow-dev` 技能更新流程：改技能仓库 → `git push` → 各环境 `git pull`。

### 环境提醒（不属于代码任务）

- [ ] 本机家目录反复变为只读（`errors=remount-ro`），会破坏 `~/.ssh` 写入与
      SSH 控制套接字。建议排查磁盘错误并恢复可写挂载。

## 3. 接手前的关键上下文

1. **构建与证据边界**：根工程默认仍是 CPU solver（MPI+METIS+CGNS）；
   `ONEFLOW_ENABLE_HIP_TESTS=ON` 才显式编译生产 HIP backend 与硬件 contract。
   `ports/kunshan/oneflow_1d_hip` 仍是独立最小闭环；root HIP contract 通过不能替代
   主 solver RK multi-stage/3D case 的 DCU 数值证据。
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
| 2026-09-19 | 把 `upstream/master` `fa3f3b06`（PR #151–#158）merge 进 `dev`：解决 upstream `FieldPipeline`/`FieldSimuRunPipeline` 重构与 dev `SimuContext` 透传的冲突——accel 状态同步并入 pipeline（INIT_FLOWFIELD 之后、Run 之前），`SolveFieldTask` 回到单一 pipeline 入口，`kInitFlowFieldTaskName` 改用集中定义的 `CmxTaskNames.h`；同步把分支模型写进 `AGENTS.md` 并校正状态文档 | commit `e584bcbb`；`master`=`origin/master`=`upstream/master`=`fa3f3b06`，`dev` 相对 upstream ahead 96 / behind 0；**本机仅完成 `g++ -fsyntax-only` 语法检查，完整构建与 fresh CPU 五算例仍待昆山验证** |
| 2026-09-16 | F2.4 3D m6 三阶段 1-step：泛化 stage runner，验证 legacy CPU、CPU batch、HIP batch 逐 stage trace；并尝试 50-step 稳定性门禁 | commits `a6ceae06`, `93955122`；昆山 DTK 26.04 / `gfx906` / `dcu:1`：896256 faces，3-stage 1-step `STAGE_TRACE_PASS`，HIP 最大差 `1.706e-13`，finite 且正状态；50-step 在 CPU/HIP 共用配置下约第 20 步共同发散，scheduler/workload 正确传播失败；不归因于 HIP 分歧 |
| 2026-09-15 | F2.3 小 case 物理/离散语义：HIP smoke 增加非 boundary-first mask、内部面守恒和 ALE/face-area 解析门禁；stage trace 增加 connectivity/geometry metadata，并从生产 face flux 重建 residual | commits `9fbbe6a7`, `fd8d9eca`；昆山 DTK 26.04 / `gfx906` / `dcu:1`：三路 residual 重建误差 `0`、守恒闭合误差 `2.804e-13`；HIP boundary `4.441e-16`、contract conservation `2.753e-14`、ALE/area `1.110e-16`；root HIP 9/9、CTest 10/10，本地根 CTest 210/210；scheduler/workload 均成功 |
| 2026-09-15 | F2.2 主 solver 小 case 1-step：新增 opt-in 多帧 face/inviscid-residual/state trace 与昆山三路 verifier；保持既有 E6 trace 兼容 | commits `156f94fc`, `2f597a04`；昆山 DTK 26.04 / `gfx906` / `dcu:1`：`plateuns2dslau2` 五方程 Lax-Friedrichs 1-step LU-SGS，HIP qf1/qf2 与 legacy 一致，invflux/residual 最大差 `1.4210854715202004e-14`，state 最大差 `6.514681130330882e-19`，finite 且正状态；root HIP 9/9、CTest 10/10；同 revision CPU 210/210、normal/strict 5/5、port 8/8；scheduler/workload 均成功 |
| 2026-09-15 | F2.1 主 solver adapter one-call：修复 HIP backend 忽略 Lax-Friedrichs `scheme=1` 与 residual kernel 假定 boundary-first 的 contract 漂移；新增显式 `boundaryMask`、3D normals、ALE mesh-normal velocity 和 face-area 覆盖 | commit `8e08c376`；昆山 DTK 26.04 / `gfx906` / `dcu:1`：257 faces × 5 equations，flux 最大差 `6.661e-16`、residual 最大差 `1.110e-15`，hardware HIP CTest 10/10；同 revision CPU 根 CTest 210/210、normal/strict 5/5、port CPU 8/8；scheduler/workload 均成功 |
| 2026-09-15 | F1 root HIP registration/guard 目标节点闭环：生产 `OneFLOW`、HIP smoke 与 root HIP contract 在 DTK 26.04 / `gfx906` / `dcu:1` 编译运行；同 revision CPU oracle 全套复验 | commits `d5005ad6`, `ca14118c`；root HIP GoogleTest/CTest 9/9；CPU 根 CTest 210/210、normal 5/5（最大 absolute difference `4.97e-10`）、strict 5/5（最大 `1.11e-17`）、port CPU 8/8；scheduler/workload 均成功 |
| 2026-09-14 | E4.4c MRField hook：新增只上传 internal cells 的 equation-major snapshot；FieldSimu 在 INIT_FLOWFIELD 后按 solver/grid key 选择 Initialize/Restart lifecycle；真实 `m6wingroe_sa` 初始化 50 步与 `startStrategy=1` restart 均通过，初始化 residual baseline 最大绝对差 `4.07e-20` | `codes/main/include/EulerDomainMrFieldAdapter.h`; `codes/main/src/EulerDomainMrFieldAdapter.cpp`; `codes/global/src/FieldSimu.cpp`; adapter 2/2、合并回归 29/29、root build 100% |
| 2026-09-15 | E6.1a/E6.2/E6.3 主 solver 细粒度验收：新增 opt-in ONEFLOW_UNS_TRACE_FILE，对 m6 3D Lax-Friedrichs legacy/batch 比较 qf1/qf2/invflux 全部 face/equation；896256 faces × 5 equations，invflux 最大绝对差 1.11e-15、最大相对差 1.41e-10；两侧 finite 且 density/pressure 为正；内部面守恒与全量 CTest 通过 | codes/uns/include/UNsInvFlux.h; codes/uns/src/UNsInvFlux.cpp; ci/kunshan/e6-cpu-trace-verify.py; ci/kunshan/e6-cpu-trace.slurm; tests/euler_cpu_adapter_test.cpp; trace verifier PASS、physicality/conservation CTest、全量 CTest 187/187 |
| 2026-09-14 | E5 RungeKutta 主 solver 接入：新增 CPU backend host stage scheduler，`TimeIntegral` 显式接收 `SimuContext` 并按 capability guard 选择 RK seam 或 legacy fallback；阶段顺序保持不变，3D 粘性 m6 case 的 RK fallback 1-step runtime 通过；无粘性改造副本因旧案例的 `nTModel=0` 空字段假设在初始化阶段退出，未作为 E5 oracle | `codes/accel/include/EulerDomain.h`; `codes/accel/src/CpuEulerDomainBackend.cpp`; `codes/main/src/EulerDomainStateSync.cpp`; `codes/multigrid/src/Multigrid.cpp`; `codes/solver/src/TimeIntegral.cpp`; CPU scheduler 4/4、capability 2/2、root build/link 100% |
| 2026-09-14 | E4.4b context-aware task seam：`ISimuTask::Execute`、`SolveFieldTask` 与 `FieldSimu` 显式传递可变 `SimuContext`；回归证明 task 可修改 owner 状态 | `codes/main/include/SimuTask.h`; `codes/main/src/SimuTaskReg.cpp`; `codes/global/include/FieldSimu.h`; `tests/main/simu_context_test.cpp`; context/task tests 17/17 |
| 2026-09-14 | E4.4a/E5 CPU state backend：生产 host state 完成 equation-major Upload/Download、CPU key 校验；E5 增加正步数 host stage scheduler，要求已 Upload 且提供 callback | `codes/accel/include/CpuEulerDomainBackend.h`; `codes/accel/src/CpuEulerDomainBackend.cpp`; `tests/euler_cpu_domain_backend_test.cpp`; scheduler 4/4 |
| 2026-09-14 | E4 lifecycle contract：初始化/restart 统一走 invalidate → create → Upload → registry insert；SimuContext 暴露 owner API；已绑定主 solver INIT_FLOWFIELD/restart 与 E5 宏步同步 | `codes/accel/include/EulerDomainStateLifecycle.h`; `codes/main/include/SimuContext.h`; `tests/euler_domain_state_lifecycle_test.cpp`; 相关 contract/context/lifecycle 测试 24/24 |
| 2026-09-14 | E6.1 主 solver CPU oracle：同一 3D m6wing Lax-Friedrichs case 分别运行 legacy 与 `ONEFLOW_ENABLE_UNS_CPU_BATCH=1` 50 步；四类结果文件通过数值比较，Slurm 完成且退出码 0 | `ci/kunshan/e3-cpu-oracle.slurm`；`aero/res/turbres` 最大差 0；`wallaero` 最大绝对差 1.0000056338554941e-12、最大相对差 2.788493125767199e-11 |
| 2026-09-14 | 集群 CPU 回归：主 solver 构建、normal/strict 五算例、port CPU contract 均通过 | 集群 `kshcnormal`；normal 5/5、strict 5/5、contract 8/8；Slurm 完成且退出码 0 |
| 2026-09-14 | E3.5 主 solver CPU batch seam：`UNsInvFlux` 在显式开关、CPU、5 方程、Lax-Friedrichs 条件下调用 adapter；新增 OneFLOW Lax-Friedrichs Roe-平均 scheme，默认路径不变 | `codes/uns/src/UNsInvFlux.cpp`; `codes/accel/src/CpuFluxBackend.cpp`; `tests/euler_cpu_adapter_test.cpp`; UNsInvFlux/CpuFluxBackend 单对象编译通过；相关测试 14/14 |
| 2026-09-14 | E2.4 registry restart 失效：新增测试证明 invalidate 后再次 GetOrCreate 会创建 fresh state，覆盖重复创建、缺失 state 与 restart 语义 | `tests/euler_domain_state_registry_test.cpp`; registry test 5/5 |
| 2026-09-14 | E3.4 residual mapping：CPU backend/adapter 支持显式 `boundaryMask`，非 boundary-first ordering 不再误用 `nBoundaryFaces`；HIP 显式 mask 已在后续 `8e08c376` 完成 | `codes/accel/src/CpuFluxBackend.cpp`; `tests/euler_cpu_adapter_test.cpp`; adapter test 5/5 |
| 2026-09-14 | E3 CPU adapter seam：新增 3/5 方程 primitive→conserved、equation-major face pack，并通过 `CpuFluxBackend` 计算 batch flux；主 solver `UNsInvFlux` 尚未接入 | `codes/accel/include/EulerCpuAdapter.h`; `codes/accel/src/EulerCpuAdapter.cpp`; `tests/euler_cpu_adapter_test.cpp`; 4/4；测试 target 编译通过 |
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
