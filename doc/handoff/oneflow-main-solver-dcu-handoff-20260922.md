# OneFLOW 主 solver HIP/DCU reconstruction residency 交接

**日期：** 2026-09-22  
**工作分支：** `dev`  
**本轮起始基线：** `aa53dc0c`（本地 `dev` 与 `origin/dev`）  
**只读基线：** `master` / `origin/master` / `upstream/master` 均为 `097c16de`  
**最终 checkpoint：** 以包含本文档的 `dev` 提交为准  
**发布边界：** F 阶段仍是 `dev` 内部工作；不创建或推进 upstream PR

## 1. 一句话结论

受限 3D 主 solver 已形成第一条连续 HIP 数据链：

```text
device q
  -> device gradient
  -> device reconstruction
  -> device qf1/qf2
  -> fused device flux/residual
```

新路径保留 CPU legacy、CPU batch 和旧 host-staged HIP fallback，并通过显式
`ONEFLOW_ENABLE_UNS_HIP_RECONSTRUCTION=1` opt-in 启用。Kunshan 上的 root HIP
构建、smoke、contract、CPU 五算例和 1-step m6 trace 已通过；fixed-CFL
50-step 的三路 workload 均正常退出且没有精确 diagnostic 命中，但临时外层
wrapper 因宽松 grep 误报而以非零状态结束，因此本轮不能把它写成标准 stability
runner PASS。

当前仍不是完整 stateful/device-resident solver：`q` 每次 gradient 仍 H2D，
gradient 仍镜像 D2H 到 MRField，residual 仍 H2D/D2H，RK/state update 仍在 host。
本轮没有做 timing，也没有更新正式性能报告。

## 2. 本轮完成的内容

### 2.1 Device ownership contract

- 新增 backend-neutral `CellFaceReconstructionView`。
- state key 继续覆盖 solver、local zone、grid level、backend/device identity。
- production HIP gradient/reconstruction 使用按 grid binding 的 backend-specific
  state；shared backend 只保留 direct/smoke fallback。
- state-owned device storage 已覆盖：
  - cell `q`，含 boundary ghost；
  - gradient `x/y/z`；
  - `qf1/qf2`；
  - face/cell geometry；
  - left/right connectivity；
  - boundary operation/mask；
  - optional face-indexed `bc_q`；
  - fused path 所需 residual scratch。
- cache 失效不再只比较裸 host pointer，同时检查 owner token、topology
  generation、extent 和 backend/device identity。
- create/reuse/invalidate/restart/teardown contract tests 已扩展。
- backend-neutral API 没有泄漏 HIP `DeviceBuffer` 类型。

### 2.2 Device reconstruction seam

HIP reconstruction 保持 CPU `ULimiter` / `UNsLimiter` 的受限语义：

- `GetQlQr` 初值来自 left/right cell state；
- 使用 face-center 到 left/right cell-center 的位移；
- 左右两侧分别使用对应 cell gradient；
- limiter-off 使用 `phi=1`；
- 保留 physicality fallback；
- boundary faces 仍位于 `[0,nBoundaryFaces)`；
- `INTERFACE` / `PERIODIC` 保留重构值；
- ordinary boundary 使用左右 cell state average；
- `SOLID_SURFACE` 用 face-indexed `bc_q` 覆盖两侧；
- boundary `rightCell` 仍指向 ghost；
- residual ownership 仍为 left `-flux`，仅 non-boundary valid right
  cell 加 `+flux`。

HIP fused flux/residual API 可直接消费 device `qf1/qf2`。NoTrace 不下载
`qf1/qf2` 或 invflux；只有 FullTrace/stage trace 显式下载诊断数组。

### 2.3 保留的 fallback 和 capability

以下路径均保留：

- legacy CPU；
- CPU batch；
- HIP gradient opt-in + host reconstruction；
- 不满足 reconstruction capability 时的既有受限 fallback/fail-fast policy。

当前 reconstruction capability 仅覆盖：

- single zone；
- finest grid；
- 5 equations；
- Lax-Friedrichs；
- limiter off；
- inviscid；
- single-rank/single-device 的当前主 solver slice。

未扩展到 limiter、viscous/turbulence、MPI/halo、multi-zone、multigrid 或 RK
algorithm change。

## 3. 数据驻留的真实状态

| 数据/阶段 | 当前状态 | 仍存在的数据移动 |
|---|---|---|
| cell `q` | state-owned device buffer 可供 gradient/reconstruction 使用 | 每次 gradient 仍从 host H2D |
| gradient `x/y/z` | kernel 写 state-owned device buffers；reconstruction 直接读取 | 仍完整 D2H 到 MRField，供 CPU oracle/diagnostic |
| geometry/connectivity | state-owned，并按 owner/generation/token 缓存 | create/rebind/invalidate 时上传 |
| boundary operation/mask/`bc_q` | state-owned；`bc_q` 按需存在 | generation 或内容失效时上传 |
| `qf1/qf2` | reconstruction 直接写 device；fused flux 直接读取 | NoTrace 无 D2H；trace 才下载 |
| invflux | fused device path 内使用 | NoTrace 无完整 D2H；trace 才下载 |
| residual | device scatter 已存在 | 当前仍有 residual H2D 和 D2H |
| RK/state update | host `UNsUpdate::UpdateFlowField` | state/residual 在 stage 边界往返 |

因此本轮真正消除的是 **device gradient → reconstruction → qf1/qf2 → fused
flux** 之间的 qf host staging；尚未消除 gradient oracle 下载和 residual/state
stage 边界往返。

本轮没有在同一正式 basis 上采集 allocation、H2D/D2H bytes、kernel launches
或 synchronize 次数，不对这些指标作下降声明。

## 4. Kunshan 证据

目标环境：

- DTK 26.04；
- HIP architecture `gfx906`；
- visible device：`dcu:1`；
- 资源 tuple 按 `ci/kunshan/README.md` 的标准 CPU/DCU 配置。

本轮 revision 在集群侧以标签 `OneFLOW-recon-p1-20260922` 保存，构建和运行证据
分别位于同标签的标准 `builds/`、`runs/` 目录；本文档不记录账号、hostname、
job ID、私有绝对路径或 raw Slurm log。

### 4.1 已通过

- root HIP configure/build；
- HIP smoke；
- HIP GoogleTest `9/9`；
- hardware CTest `10/10`；
- CPU normal `1e-8` 五算例 `5/5`；
- CPU strict `1e-15` 五算例 `5/5`；
- CPU regression scheduler `COMPLETED/0:0`；
- 1-step m6：
  - legacy CPU workload exit `0`；
  - CPU batch workload exit `0`；
  - HIP gradient + reconstruction workload exit `0`；
  - stream verifier `STAGE_TRACE_PASS records=12`；
  - HIP 相对 legacy 最大绝对误差约 `2.3981e-13`；
  - qf1/qf2、invflux、residual、state、finite、positive
    density/pressure、connectivity/geometry metadata 均通过。

### 4.2 仍需标准 runner 闭环

fixed-CFL `0.01`、50-step 直接 workload：

- legacy / CPU batch / HIP reconstruction 均 exit `0`；
- 三路精确 diagnostic regex 均为 `diagnostic_hits=0`。

临时 wrapper 使用宽松 grep，将普通日志中的 `information` 误判为 diagnostic，
导致外层 Slurm 状态为失败。真实 workload 没有失败，但 scheduler/workload 的标准
双重判据没有在同一个正确 runner 中闭环。明天应先修正/复核 runner 的精确匹配，
再在最终提交 revision 上重跑；在此之前不要写成标准 50-step PASS。

### 4.3 性能边界

- 本轮没有 timing。
- 既有 gradient slice 只有一次正式同-basis 作业：
  - legacy CPU：`14927.309397 ms`；
  - CPU batch：`15519.090773 ms`；
  - HIP gradient：`12866.935637 ms`；
  - HIP/legacy：`1.160129x`。
- 该数据不能被描述为稳定约 16% 加速，也不能外推到本轮 reconstruction seam。
- 不更新正式性能报告。

## 5. 明天的严格执行顺序

### P0：先在最终 checkpoint 上补齐标准稳定性证据

1. 确认 `dev == origin/dev`，工作区为空。
2. 使用最终提交生成新的 cluster source archive，不复用未提交源码。
3. 修正或复核 stability wrapper，只使用标准精确 diagnostic regex。
4. 重跑 fixed-CFL `0.01`、50-step：
   - legacy；
   - CPU batch；
   - HIP gradient + reconstruction。
5. 记录 workload exit、diagnostic hits 和 scheduler state；三者必须同时成立。
6. 若 runner 失败，先区分 workload 数值失败、HIP runtime 失败和 wrapper
   解析失败，不用 scheduler state 代替 workload 证据。

### P1：补 reconstruction/boundary 小 case contract

优先补齐：

- interface boundary preserve；
- periodic boundary preserve；
- solid-surface face-indexed `bc_q` override；
- ordinary boundary average；
- ghost gradient copy；
- qf1/qf2 FullTrace 与 NoTrace 数据移动 contract；
- generation/token 变化后的 geometry、boundary 和 `bc_q` cache invalidation。

继续比较 qf1、qf2、invflux、residual、state 和 connectivity/geometry metadata。

### P2：消除无条件 gradient D2H

1. 让 device gradient view 成为 HIP reconstruction 的生产输入。
2. 将 MRField gradient 下载改为明确的 CPU oracle/FullTrace/diagnostic fallback，
   不在 NoTrace 每 stage 无条件执行。
3. 记录改变前后的：
   - allocation count；
   - H2D/D2H bytes；
   - kernel launches；
   - synchronize count；
   - per-stage wall-clock。
4. 不用 async API 掩盖未定义依赖；先保持明确 stage fence。

### P3：迁移 residual/state update

1. 只读确认
   `LOAD_RESIDUALS → UPDATE_RESIDUALS → CALC_LHS → UPDATE_FLOWFIELD`
   的真实数据契约。
2. 在现有受限 capability 下增加：
   - device residual zero/accumulate；
   - residual scaling/LHS；
   - device q update。
3. 评估 `UNsUpdate::UpdateFlowField` host cell loop 的 HIP kernel 化。
4. 保留 host path 和 capability fail-fast。
5. 不同时迁移 viscous/turbulence、MPI halo 或改变 RK algorithm。

### P4：主链稳定后再做的工作

- stage-level stream/event 优化；
- residual scatter 的 atomic / CSR / segmented reduction 比较；
- device-side finite/positivity/error/conservation reduction；
- MPI/halo 与 GPU-aware MPI capability probe；
- 多次同 basis timing。

## 6. 明天接手时的检查清单

开始前阅读：

1. `AGENTS.md`
2. `doc/plans/oneflow-development-todo.md`
3. `ci/kunshan/README.md`
4. `doc/plans/oneflow-euler-optimization-plan.md`
5. `scientific-dcu-porting` skill 与 `references/porting-playbook.md`
6. 本文档

只读检查：

```bash
git status --short --branch
git log -2 --oneline
git rev-parse dev origin/dev master origin/master upstream/master
git diff --check
```

集群验证继续使用 `ci/kunshan/f2-main-solver-stage.slurm` 和标准 workspace layout。
本机不重复完整 build；除非为定位纯编译问题，构建和数值验证放到 Kunshan 目标节点。

## 7. 禁止事项

- 不在 `master` 开发。
- 不主动创建或推进 upstream PR。
- 不删除 CPU legacy/batch 或旧 HIP fallback。
- 不把 capability 扩大到 limiter、viscous/turbulence、MPI、multi-zone 或
  multigrid。
- 不把 1D Euler stateful benchmark speedup 外推到 3D 主 solver。
- 不从 CUDA/ROCm/Kokkos 文档推断 DTK 能力。
- 不把一次 timing 写成稳定性能结论。
- accuracy、stability、MPI correctness 和多次同 basis timing 未全部通过前，
  不更新正式性能报告。

## 8. 本轮收尾状态

- 共享代码与 fork-only handoff/TODO 分开提交，便于将来按发布边界处理。
- 本项目的 ignored case logs 和过期本地草稿在收尾时清理；受版本控制的 fixture
  不删除。
- 最终应满足：

```text
dev == origin/dev
working tree clean
master/origin/master/upstream/master unchanged
```
