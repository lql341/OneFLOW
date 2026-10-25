# OneFLOW 长期分阶段开发计划（accelerator 主线）

> 最后更新：2026-10-25。本文是长期路线图（living），按季度/阶段维护；
> 每一轮的细碎任务、证据和 A/B 数据仍记在
> [`oneflow-development-todo.md`](oneflow-development-todo.md) 与 `doc/handoff/`。
> 规则以根目录 `AGENTS.md` 为准；集群流程见 `ci/kunshan/README.md`。

## 0. 结论先行

OneFLOW 的加速主线已经证明了两件事，长期计划的形状由它们决定：

1. **执行模型比 kernel 重要一到两个数量级。**
   1D standalone 端口：`advance` 内循环（设备驻留、每步 2 次同步）
   CPU 1 线程 vs 1 DCU 为 `102.8x/137.2x/139.1x/155.2x`（65k→4M cells）；
   同一套 kernel 走 trace/host-staged 路径只有 `2.0–2.6x`；
   应用级 lifecycle（含网格、初始化、IO、MPI）相对 32 rank CPU 为
   `3.85–9.70x`，4 卡/4M 为 `25.55x`。
   三者差两个数量级，差别全在"谁拥有状态、每步回几次 host"。
2. **3D 主求解器还在第二阶段。** 现在每步 `31.6 ms`（8 zone/8 rank/1 DCU，
   paired 口径），其中 device kernel 只占约 `1–1.5 ms`，其余是 host task
   调度、stage 间 fence、halo 往返和仍留在 host 的整场拷贝。
   也就是说 3D 目前处在 1D 的"trace 路径"位置，而不是"驻留执行"位置。

因此长期计划的主轴不是继续抠 kernel，而是：

> **Phase A 把 3D 改成设备驻留执行（device-resident Advance）→
> Phase B 补齐数值能力（scheme/limiter/粘性/SA/隐式/multigrid）→
> Phase C 真实算例与 IO → Phase D 多卡多节点 →
> Phase E 架构通用化 → Phase F 产品化与发布。**

每个阶段都有"准入门"（oracle/门禁）和"退出门"（可发布证据），见 §3。

## 1. 代码架构现状（长期计划的事实基础）

### 1.1 分层与规模

| 层 | 目录 | 规模（cpp/hip + h） | 职责 | 设备状态 |
|---|---|---:|---|---|
| 入口/调度 | `codes/main`（1.9k）、`codes/solver`（3.8k）、`codes/task`（3.9k）、`codes/register`（1.3k） | ~11k | `Simulation → SimuImp → FieldSimu → Multigrid → TimeIntegral`，task 字符串注册与派发 | host |
| 数据/字段 | `codes/database`（2.8k）、`codes/global`（4.8k）、`codes/adt`、`codes/array` | ~9k | `DataBase`/`DataBook`、`MRField`、`FieldHome`/`FieldManager`、变量名映射 | host（设备侧另有影子缓冲） |
| 网格/几何 | `codes/geometry`（18.4k）、`codes/zone`（1.1k）、`codes/partition`（1.5k）、`codes/cgns`（8.7k） | ~30k | 拓扑、面-单元关系、BC 记录、METIS 分区、CGNS IO、wall distance | host |
| 物理/离散 | `codes/ns`（4.9k）、`codes/uns`（4.7k）、`codes/ins`（3.0k）、`codes/uins`（4.0k）、`codes/turb`（3.9k）、`codes/uturb`（3.6k）、`codes/lhs`、`codes/grad`（+`usolver`/`uns` 侧） | ~28k | NS/INS/湍流三族 solver 的 RHS、边界、LUSGS、梯度、通量 | 受限 3D inviscid 链已 device |
| 时间/迭代 | `codes/time`、`codes/multigrid`（0.9k）、`codes/unsteady`（0.9k）、`codes/implicit` | ~3k | 显式 RK、LUSGS、多重网格、双时间 | host |
| 加速器 | `codes/accel`（10.6k）、`codes/cuda`（0.36k） | ~11k | `AccelRuntime/AccelBackend/DeviceBuffer`、`FluxBackend`、`EulerDomain`、`ResidentStateContract`、`StageProfiler`、HIP kernel | **唯一 device 拥有者** |
| 后处理/IO | `codes/restart`（0.6k）、`codes/visual`（3.4k）、`codes/force`（1.0k）、`codes/residual`（0.8k）、`codes/io`（2.9k）、`codes/postprocess`（0.9k） | ~10k | restart、可视化、气动力/Cp、残差、文件 | host |
| 独立端口 | `ports/kunshan/oneflow_1d_hip`、`codes/scalar`（8.9k） | — | 1D Euler 驻留执行样板；scalar toy path | device |
| 网格生成 | `codes/special`（11.2k） | 11k | 块结构网格生成、曲线/曲面 | host，**不在加速范围** |
| 测试/CI | `tests/`、`test/`、`ci/kunshan/`、`tools/oneflow_gui` | — | GoogleTest 契约、五算例回归、多区 trace/XB/稳定性 runner、GUI | — |

### 1.2 真实执行链

```text
Simulation::Run
  → SimuImp::PreProcess            # MPI/设备/控制参数
  → FieldSimu
      → LoadGrid / Partition / CreateSolvers / INIT_FLOWFIELD
      → MultigridSolve             # 外层迭代 + grid level
          → TimeIntegral::Relaxation
              → RungeKutta / Lusgs / Simple   # host task 序列
                  LOAD_Q → CALC_TIME_STEP
                  [每个 stage] LOAD_RESIDUALS → UPDATE_RESIDUALS
                               → CALC_LHS → UPDATE_FLOWFIELD → CALC_BOUNDARY
                               → UPDATE_INTERFACE_DATA
                  post: UPLOAD/UPDATE/DOWNLOAD_INTERFACE, 残差/气动力/输出
  → SimuImp::PostProcess           # 收尾、restart、teardown
```

`UPDATE_RESIDUALS` 内部：`NsCalcRHS = inviscid flux（梯度/limiter/重构/Riemann）+ viscous flux + source`；
task 通过全局 `ug / unsf / nscom / ctrl / ZoneState::zid` 隐式定位到"当前 zone/grid"。

### 1.3 三个并存的"世界"

| 世界 | 触发 | 状态所有者 | 数值路径 | 用途 |
|---|---|---|---|---|
| legacy host | 默认 | 全局 `MRField` | 逐面/逐 cell host 循环 | CPU oracle、全部未迁移能力 |
| CPU batch | `ONEFLOW_ENABLE_UNS_CPU_BATCH=1` | 同上 + `EulerDomain` CPU state | 批量 adapter 调 `CpuFluxBackend` | 批量路径的等价性验证 |
| HIP device | `ONEFLOW_ACCEL_BACKEND=HIP` + capability | `Ns3DDeviceState` + `ResidentStateContract` | 每 stage 一批 kernel，host 调度 | 受限 3D inviscid 快路径 |

三者靠 `EulerInvFluxCapability`、`EulerRungeKuttaCapability`、`HipProductionPolicy`
准入，靠 3 步多区 trace（1e-10）+ 200 步跨后端残差（1e-12）+ CPU 五算例（1e-8/1e-15）
保证等价。

### 1.4 已有 seam（可复用的骨架）

| seam | 位置 | 现状 |
|---|---|---|
| `AccelRuntime` / `AccelBackend` / `DeviceBuffer` | `codes/accel` | 成熟：设备选择、拷贝、同步、RAII |
| `EulerDomainBackend::{CreateState,Upload,Advance,Download}` + `StateRegistry` | `codes/accel/include/EulerDomain.h` | **3D 只接了 CPU**；HIP 没有 `Advance` |
| `EulerRungeKuttaCapability`（含 `stageCallback`） | `codes/solver` | 已完成接入点：`TimeIntegral::RungeKutta(context)` 里 `backend.Advance(state,1,options)` |
| `EulerInvFluxCapability` / `HipProductionPolicy` | `codes/accel` | 逐 stage 准入 + 回退 |
| `ResidentStateContract` | `codes/accel` | 设备/主机数据所有权与代际（本轮新增 host-mirror currency） |
| `StageProfiler` | `codes/accel` | opt-in 分段计时（含 host 子阶段） |
| 1D standalone 驻留端口 | `ports/kunshan/oneflow_1d_hip` | 已证明驻留执行模型 |
| `FluxBackend`（face flux / residual） | `codes/accel` | 只覆盖 face 级操作，不足以承载 stage |

### 1.5 结构性问题（长期计划要逐个消除）

1. **ambient globals 决定热路径**：`ug/unsf/nscom/ctrl/Iteration/ZoneState::zid`
   让"哪个 zone 的哪份 device state 是当前的"靠隐式约定；本轮 device-authoritative
   工作反复踩到（初始化上传、`ug.Init()` 重绑定、contract 失效）。
2. **host 调度 + 逐 stage kernel**：每个 stage 至少一次 kernel fence、一次 task 派发、
   若干次 host 侧整场循环；这正是 3D 目前"31.6 ms 里 kernel 只占 1.5 ms"的原因。
3. **capability 判断分散**：`HipProductionPolicy` 的开关 + 各文件里的 `UseHip*`
   守卫 + 两个 capability 枚举；能力扩展时会组合爆炸。
4. **双实现 + 只有 trace 级等价**：host 路径是唯一 oracle，device 路径没有单元级
   数值契约（每加一种 scheme/limiter 都要靠整链路 trace 判定）。
5. **fixture 稀缺**：唯一合法可分区网格是 8 zone m6wing（294,912 cell）；
   `partition_type=0`(GenerateLink) 仍失败；≥1M cell 的多卡结论无法产出。
6. **口径/文档债**：3D 性能从未发布为正式报告（现有 `performance/` 都是 1D 端口）；
   `oneflow-development-todo.md` 的摘要仍停在 2026-10-23 的 `3.89x`。

## 2. 阶段划分

依赖关系：

```text
Phase A（执行模型）
   ├── Phase B（数值能力）── Phase C（真实算例/IO）
   └── Phase D（多卡）──────┘
                 Phase E（架构通用化）, 与 A/B 并行、按痛点触发
                        Phase F（产品化/发布），每个阶段末都要做一遍
```

### Phase A — 3D 设备驻留执行（下一个迭代，最高优先）

目标：让 3D 显式路径从"host 调度 + 逐 stage kernel"变成"一个宏步一次进入设备"，
把 host 往返和 fence 从每步几十次降到 ≤2 次。

| 子项 | 内容 | 预期 | 证据/入口 |
|---|---|---|---|
| A1 | 固定开销三刀：gradient 镜像只在最后 stage、设备路径跳过 `LOAD_Q`/`LOAD_RESIDUALS` 整场 host 拷贝、基准 warmup 修正（`bnd_resize` malloc 假象） | 31.6 → ~26 ms | `UNsInvFlux::CalcInvFace`、`FieldTaskReg::LoadQ/LoadResiduals`、`HipFluxBackend::UpdateBoundaryState` |
| A2 | 每步一次 halo：3 stage + 1 post 的 4 轮合并；梯度/q 接收同轮；DeviceDirect 或 GPU-aware | −2.8 ms | `InterfaceTask`、`UNsGrad::CalcGradHip`、`ExchangeHipInterfaceQ` |
| A3 | stage kernel 融合：gradient→recon→flux→lhs→update→boundary 在设备内串联，stage 内不再有 host 同步点 | −6 ms | `HipFluxBackend` 各 kernel + `TimeIntegral` |
| A4 | **HIP `EulerDomainBackend::Advance`**：把 RK stage 循环搬进 backend（复用已有 seam 与 `stageCallback`），host 只负责 INIT/输出/门控；每宏步结束按需 Download | 达到 12–14 ms | `TimeIntegral::RungeKutta(context)`、`EulerDomain.h`、`HipSmoke` 既有 `Advance` 样板 |

退出门：
- 8 zone/8 rank/1 DCU 1000 步 paired **≤14 ms/step**（相对 10-25 的 31.6 再快 2.2x，
  相对 10-24 的 38.3 快 2.7x）；
- 3 步多区 trace（1e-10）、200 步跨后端（1e-12）、CPU 五算例（1e-8/1e-15）、
  50 步稳定性全绿；
- 每宏步 host 往返次数与 H2D/D2H 字节数进入 `StageProfiler` 报告。

### Phase B — 数值能力（决定这条后端是否有实用价值）

现状：设备路径只接受 `MULTI_STAGE + Lax-Friedrichs(5) + ilim=0 + vismodel=0`。
仓库自带的旗舰算例 `test/cfd/m6wingroe_sa` 是 `LU-SGS + Roe + SA`，**完全在 host**。

| 子项 | 内容 | 依赖 | 关键文件 |
|---|---|---|---|
| B1 | 无粘通量 scheme：LF → Roe / HLLC / AUSM 系列（device），LF 保留作 oracle | A | `UNsInvFlux`、`NsInvFlux`、`HipFluxBackend` flux kernel |
| B2 | limiter：Barth / Venkat（cell 邻域极值），解除 `ilim=0` | B1 | `UNsLimiter`/`NsLimiter`、recon kernel |
| B3 | 层流粘性：`UNsVisFlux`（面粘性通量、T 梯度、壁面热流） | B1/B2 | `UNsVisFlux`、gradient（多方程）、BC 热流 |
| B4 | 湍流：SA（1 方程）优先，SST 可选；`turb` solver 与主 solver 的 task 交互 | B3 | `codes/turb`、`codes/uturb`、`system/turb` 字段 |
| B5 | 隐式：LU-SGS 设备化（下/上扫、`dq` 交换、与 halo 的依赖），再评估双时间 | A4 | `NsLusgs`、`ULhs`、`LhsTaskReg` |
| B6 | 多重网格：restrict/prolong 的设备化，或显式声明"只加速 finest grid，粗网格留 host" | A4 | `codes/multigrid` |

准入门（每个子项都必须先过）：
1. 单点数值契约（scheme/limiter 的 CPU vs device 单元测试，≤1e-14）；
2. 3 步多区 trace（1e-10）；
3. 该能力对应的物理门禁（正性、守恒、边界语义）；
4. 200 步跨后端 + 50 步稳定性。

退出门：`m6wingroe_sa`（Roe+LU-SGS+SA）在设备上跑通并通过同一组门禁；
之后才允许把 "HIP 支持的算例范围" 写进正式文档。

### Phase C — 真实算例与 IO（把加速变成可交付能力）

| 子项 | 内容 | 说明 |
|---|---|---|
| C1 | IO/restart：设备输出或"整场一次性 D2H + 异步写盘" | 现在每输出步 `NsRefreshHostStateMirror` 整场回拉，长跑/多输出会吃掉收益 |
| C2 | 气动力/Cp/热流：设备侧面量归约，或维持边界 cell 级小回拉 | 现在每步一次边界 cell 镜像，可接受但需量化 |
| C3 | 残差/收敛判据设备归约 | 已在设备侧算 metrics，需扩到多 zone/多 solver 聚合 |
| C4 | 内存预算与大网格 | 记录每 100 万 cell 的 device 常驻字节数，给出单卡可解规模 |
| C5 | 定常/非定常的真实工况回归 | 与 `test/suites` 对齐，形成"设备算例集" |

退出门：≥1M cell 的单卡算例端到端可跑（含 restart 等价、气动力对比、可视化），
并有对外可引用的 paired 性能数字。

### Phase D — 多卡与多节点（结论的前提是 fixture）

| 子项 | 内容 |
|---|---|
| D1 | **fixture**：拿到 ≥1M cell 且带 `.skl` 的合法分区网格，或修 `GenerateLink`（`codes/geometry/src/FaceTopo.cpp` 的 `SetNewFace2Node`）建表 |
| D2 | 设备 halo：GPU-aware MPI（1D 已验证 `1.39/1.31/1.19/1.13x`）在 3D 的复现；DeviceDirect 在 packed layout 上重测 |
| D3 | 映射与负载：1 rank = 1 zone 的约束是否长期成立；需要 zone 切分/多 zone per rank 时的 state 与 halo 语义 |
| D4 | 评测口径：固定 per-rank cell 数、同一 repeats、1 DCU 槽 = 8 核；给出 8:1、32:1 的合法性说明 |

退出门：至少 4 卡、per-rank 恒定 cell 数的端到端数字，且与同节点 CPU 同核数配对。

### Phase E — 架构通用化（与 A/B 并行，按痛点触发）

| 子项 | 内容 |
|---|---|
| E1 | 热路径去 ambient globals：把 zone/grid/solver 身份显式传进 stage（先 device 路径，再 host） |
| E2 | capability 表驱动：`solver × scheme × limiter × viscous × time-integral × grid-level × halo` 一张表 + 单测，替换散落守卫 |
| E3 | `EulerDomain` 泛化到 NS：3/5 方程已支持，补源项/粘性/能量扩展与 view 元数据 |
| E4 | backend-kind 分派：清掉 `IsAccelerator()` 当 HIP 用的地方，为 CUDA/Kokkos 留真实入口 |
| E5 | 清理 toy/scalar 路径与失效文档，保持 CPU oracle 单一 |

退出门：新增一种 scheme 或一种物理项，只需"新 kernel + 表项 + 单测"，不需要改
`TimeIntegral`、`UNsInvFlux`、`InterfaceTask` 的骨架。

### Phase F — 产品化与发布（每个阶段末循环一次）

| 子项 | 内容 |
|---|---|
| F1 | 3D 正式性能报告（Markdown + HTML 同 stem）：paired、同节点、同 repeats、含环境口径 |
| F2 | CI 矩阵：CPU normal/strict、DCU 单卡、DCU 多卡、nightly 长跑；PR 分支自跑门禁 |
| F3 | 文档：`doc/plans/oneflow-development-todo.md` 摘要与 §2 现状、`ci/kunshan/README.md`、Sphinx 教程 |
| F4 | 交付：GUI/示例算例、发行说明、对外能力矩阵（哪些 solver/scheme 在设备上已验证） |

## 3. 里程碑与判定口径

| 里程碑 | 目标（8 zone/8 rank/1 DCU，1000 步，paired） | 能力 | 判定 |
|---|---|---|---|
| M0（已达成，2026-10-25） | 31.6 ms/step（≈6.4x @8:1） | 稳态 + inviscid + LF + ilim=0 | trace 24/0/0、XB 2.33e-18、CPU 5/5、PR 已合并 |
| M1（Phase A） | ≤14 ms/step（≈14–16x） | 同上，但宏步驻留 | A 退出门全部通过 |
| M2（Phase B） | 保留 M1 性能的 70% 以上 | Roe/HLLC、limiter、层流粘性、SA | `m6wingroe_sa` 设备端通过门禁 |
| M3（Phase C） | 1M cell 单卡可交付 | + IO/restart/气动力 | 设备算例集 + 正式报告 |
| M4（Phase D） | 4 卡 ≥3x 单卡（per-rank 恒定） | 多卡 halo | 合法 fixture + 配对数据 |
| M5（Phase E/F） | 不追性能 | 通用化 + 发布 | 新能力只需表项 + kernel |

口径纪律（沿用 `AGENTS.md`）：
- 只比同节点、同 repeats、同 fixture、同 rank/资源；`lifecycle_*_ms` 是 repeats 总和；
- 不以 100 步或 kernel-only 作生产结论；1 DCU 槽 = 8 核；
- 探索性单作业数据只进 todo，不进正式报告。

## 4. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 双实现漂移（host oracle 与 device 结果缓慢分叉） | 数值不可信 | 每个能力先做单元级数值契约；trace/XB/五算例是准入门而不是事后验证 |
| ambient globals 阻碍驻留执行 | Phase A4 无法干净落地 | E1 提前做 device 路径的显式上下文；host 侧渐进 |
| fixture 缺（≥1M 合法分区） | Phase D 无法收口 | 现在就去谈 owner；同时评估修 `GenerateLink` 的代价 |
| 隐式/multigrid 复杂度高 | 工期失控 | 先把显式驻留做到 14 ms；隐式与 multigrid 各自独立立项、独立门禁 |
| capability 组合爆炸 | 维护成本 | E2 表驱动 + 单测；每个组合要么有门禁要么默认拒绝 |
| 性能口径漂移（跨会话跨节点比较） | 结论反复 | 只发 paired；报告里写清节点/会话限制 |
| 上游同步与 PR 规模 | 评审不可行 | 每轮三端同步 master、按"backend/stage/CI"小切片提 PR（#227 的教训） |

## 5. 与既有文档的关系

| 文档 | 已有结论 | 本文新增 |
|---|---|---|
| `doc/plans/oneflow-development-todo.md` | A–G 分层（A–E 已闭环、F 收口、G 进行中）；P0–P3 任务清单与每轮证据 | 把 G 展开为 A–F 六个阶段，给出准入门/退出门与里程碑 |
| `doc/architecture/oneflow-1d-euler-dcu-architecture-audit-20260904.md` | backend-neutral 接入方案、state key、调用边界 | 把该方案升格为长期阶段主轴（Phase A4） |
| `doc/plans/oneflow-euler-optimization-plan.md` | "stateful NoTrace 才是性能路径"（trace 2x vs 驻留 100x+） | 在 3D 复现同一结论：执行模型优先于 kernel |
| `doc/runbooks/oneflow-dcu-porting-playbook-20260903.md` | 移植方法论（契约→或修→门禁→benchmark） | 作为每个 Phase 子项的验收流程模板 |
| `doc/handoff/oneflow-3d-hip-handoff-20261025.md` | 本轮 device-authoritative q 的状态、坑与下一步 | 其 P0.1–P0.4 即本文 Phase A 的 A1–A3 |

## 6. 需要拍板的开放问题

1. **目标规模**：这条后端的"完成"定义是 1M cell 单卡、还是 100M cell 多卡？
   （决定 Phase C/D 的投入比例）
2. **旗舰算例优先还是性能优先**：先做 Phase B（Roe+SA+LU-SGS，让 `m6wingroe_sa`
   上设备），还是先把 Phase A 做到 14 ms 再扩能力？
3. **隐式与多重网格是否在设备上做**：若只加速显式 finest grid，可省掉 Phase B5/B6
   的大半工作，但要接受"定常收敛用 LUSGS 就没加速"的现实。
4. **zone 映射**：长期是否维持 1 rank = 1 zone？若需要 zone 切分，Phase D 的工作量
   翻倍（state、halo、load balance 全变）。
5. **CUDA/Kokkos 是否真要做**：做则 E4 要在 Phase A 之前，避免把 HIP 细节写进 seam。

## 7. 立刻可执行的三件事（两周内）

1. **Phase A1**（3 刀，31.6 → ~26 ms）+ paired 测量 + 全门禁。
2. **3D 正式性能报告**：同节点补 CPU 8C 1000 步 2–3 个样本，与 HIP 配对，
   新建 `doc/reports/performance/oneflow-3d-...`，Markdown 与 HTML 同步；
   同时把 `oneflow-development-todo.md` §1 摘要改成当前事实。
3. **fixture 决策**：向 fixture owner 申请 ≥1M cell 带 `.skl` 的 m6wing 级网格；
   并行评估 `GenerateLink`（`FaceTopo::SetNewFace2Node`）修复代价，决定是否自研。
