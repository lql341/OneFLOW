# OneFLOW 开发待办与衔接（living document）

> 最后更新：2026-09-22（完成 reconstruction/residency contract 只读盘点；下一步先建立 state-owned device contract，3D 仍不是完整 stateful/device-resident）
> 用途：每轮任务开始前读本文档，结束后更新本文档。让任何人或智能体
> 接手时只读这一份就能继续推进。
>
> 配套入口：[`AGENTS.md`](../../../AGENTS.md)（规则与文档地图）、
> [`ci/kunshan/README.md`](../../../ci/kunshan/README.md)（集群流程与标准套件）。

## 交接摘要（先读这一段）

**现在在飞什么**

| 项 | 状态 |
|---|---|
| `origin/dev` / 本地 `dev` | 已同步到当前 checkpoint；`dev` 已合入 `upstream/master`，并包含主线重复定义修复与 3D HIP residual 接入。精确值以 `git rev-parse dev` 为准，不要在本文档里钉死自己的 commit |
| PR #159 `pr/agents-branch-model` | OPEN / MERGEABLE；1 文件 +23/−0；CI 4/4 绿；内容为 `AGENTS.md` 的 fork 无关分支模型。**纯文档**，不涉及数值/后端，规则 1 的五算例与 HIP contract 均不适用 |
| PR #160 `pr/euler-weno5-unified` | OPEN / MERGEABLE；64 文件 +4771/−272；CI 4/4 绿；内容为 accelerator substrate + CPU vertical slice + 1D Euler port 的 WENO5/HIP contract |

两个 PR 都从 `upstream/master` 分 topic 分支开出，未包含 fork-only 文档。**#160 已实跑规则 1 的两个门禁**（昆山 T1 + T2，见 §1）；#159 是 docs-only，按规则 1 的适用范围不需要门禁。

**当前状态**：首个 gradient device migration vertical slice 已完成。`ONEFLOW_ENABLE_UNS_HIP_GRADIENT=1` 仅在单 zone、finest grid、5 方程、Lax-Friedrichs、limiter off、inviscid 的 HIP batch 路径启用 Green–Gauss gradient；geometry/connectivity 可缓存，但每次 gradient 仍从 host 上传 primitive state，并把 `dqdx/dqdy/dqdz` 下载回 host。Kunshan 的 root HIP build/smoke/contract、3-step accuracy、fixed-CFL 50-step stability 和同 basis timing 均通过；HIP `gradient` stage 从此前约 `1046.610 ms` 降为本次 `176.883 ms`，单作业端到端 HIP/legacy 为 `1.160129x`。该结果仍是一次正式作业，不更新已发布性能报告；3D 仍为 host-staged，远未达到 1D Euler 的完整 stateful/device-resident 状态。

### 2026-09-22 新会话 handoff

**收益判断**：gradient kernel/stage 迁移是有效的，但还没有形成显著、可稳定声明的应用级加速。当前同 basis 只有一个正式性能作业，HIP/legacy 为 `1.160129x`；此前未启用 device gradient 的单作业为 `1.085759x`，两次作业之间不能当作严格配对重复。现有结果只说明方向正确，不足以宣称稳定约 `16%` 加速。

当前 HIP breakdown 中 `reconstruction=933.296 ms`，占 HIP 总 wall-clock `12866.936 ms` 约 `7.25%`。即使孤立地把 reconstruction 时间完全消除、其他阶段不变，当前 basis 的理论额外 speedup 上限也只有约 `1.08x`；实际还会有 kernel、同步和数据搬运开销。因此若目标是显著性能提升，不能只继续逐个替换 host kernel，必须逐步形成 `q → gradient → reconstruction → flux/residual` 的连续 device-resident 数据链。

**新会话第一件事（只读，不先写 kernel）**：盘点 limiter-off reconstruction 的调用链与 MRField 数据契约，同时回答以下问题：

1. reconstruction 输入是否可直接消费 device gradient，避免本轮 `dqdx/dqdy/dqdz` D2H；
2. reconstructed `qf1/qf2` 是否可直接留在 device，供 `CalcAndAddPrimitiveFaceFlux` 消费，避免下一次 H2D；
3. boundary reconstruction 的顺序、ghost extent、face ownership 和 trace 语义如何保持与 CPU oracle 完全一致；
4. 哪些 buffers 应迁入按 solver/zone/grid/backend key 管理的 `Ns3DDeviceState`，而不是继续由 process-wide shared backend 隐式持有；
5. 如果最小 reconstruction slice 仍要求 gradient D2H + face-state H2D，则先不要把它作为性能优化实现；只在它能建立连续驻留链或明确架构价值时继续。

**建议决策**：若下一轮目标是架构推进，可实现受限的 device reconstruction vertical slice；若目标是明显加速，应优先设计 persistent state/device ownership 与跨 stage 数据驻留，再把 gradient/reconstruction 串起来。两条路线都必须保留 CPU oracle，且不同时扩展 limiter、RK、viscous/turbulence 或 MPI。

**下一步（按优先级）**

1. PR #159 / #160 暂不主动推进；只在收到 review 反馈时处理，并在对应 topic 分支重跑门禁。
2. fixed-CFL 50-step 稳定性已通过；旧 v5b 的共同负压来自 runner 保留 `cfled=10` 的 CFL ramp，后续不再把该快照当作 HIP blocker。
3. 先完成 reconstruction/residency contract 只读盘点；不要直接写一个仍需 gradient D2H 和 qf H2D 的孤立 kernel。
4. 若选择实现 reconstruction slice，仍按 CPU oracle/trace → 3-step accuracy → fixed-CFL 50-step stability → 同 basis timing 验收，不一次扩大到 limiter/RK/viscous。
5. 优先把 shared backend buffers 收敛到 `Ns3DDeviceState` ownership，并逐步消除 q H2D、gradient D2H 和 reconstructed face-state H2D。
6. 完成完整 NS 与 MPI correctness；只有连续多次同 basis 测量方向一致且 accuracy/stability/MPI 全通过后，才更新已发布性能报告或整理后续 PR。

**注意**：dev 上仍留有大量未上游内容（F 阶段 DCU 主 solver 那批），它们**未收口、不要提前提 PR**；提 PR 的三个坑见 §协作约定。

**如何复现本轮门禁**（`$W` = 集群工作区根，具体绝对路径在集群侧 `README.md`）

```bash
# 0) 把要验证的 revision 打包上传（必须是 git archive，不是 clone）
R=<短sha>; git archive --format=tar.gz --prefix="OneFLOW-$R/" <ref> -o /tmp/OneFLOW-$R.tar.gz
scp /tmp/OneFLOW-$R.tar.gz kseshell:/tmp/
ssh kseshell "rm -rf \$W/src/OneFLOW-$R; tar -xzf /tmp/OneFLOW-$R.tar.gz -C \$W/src"

# 1) T1 cpu-regression（kshcnormal，16 CPU）——脚本按 $W/scripts/ 下的参数化版本
ssh kseshell "sbatch --export=ALL,REV=$R \$W/scripts/oneflow-cpu-regression-param.slurm"

# 2) T2 dcu-single（kshdnormal，dcu:1）——必须用**带 amd_comgr_DIR 修复**的脚本
ssh kseshell "sbatch --time=01:00:00 \
  --export=ALL,ONEFLOW_EULER_SOURCE_DIR=\$W/src/OneFLOW-$R/ports/kunshan/oneflow_1d_hip,\
ONEFLOW_BUILD_DIR=\$W/builds/port-dcu-$R,\
ONEFLOW_ARTIFACT_DIR=\$W/runs/<date>/dcu-single-$R/artifacts \
  \$W/src/OneFLOW-<含修复的分支>/ci/kunshan/euler-dcu-gtest.slurm"
```

- 产物落在 `runs/<date>/<suite>-<rev>/artifacts/`：`result.txt`、`exitcodes.txt`、`normal.log`、`strict.log`、`gtest.log`、`ctest.log`。
- 判据：T1 为 `CPU_REGRESSION_STANDARD_PASS` 且两档各 5/5；T2 为 `config=0 build=0 test=0` 且 GoogleTest/CTest 各 9/9。
- **坑**：upstream 旧版 `euler-dcu-gtest.slurm` 在 `module purge` 之前解析 cmake 且不传 `-Damd_comgr_DIR`，在集群上必失败；本轮用的是 dev 上已修好的副本，PR #160 已把同一修复带入。此外 cmake 模块在部分节点加载不稳定，必要时显式指定 cmake 3.25 路径。

### 2026-09-22 reconstruction/residency contract 只读盘点

本轮先按 handoff 要求完成调用链和数据契约盘点，**没有新增 kernel，也没有把仍需
gradient D2H + qf H2D 的路径包装成性能优化**。结论如下：

1. **reconstruction 输入不能直接消费当前 device gradient。** `UNsInvFlux::CalcInvFace`
   先执行 `UNsGrad::CalcGradHip`，`HipFluxBackend::CalcGradient` 把 `q` 的
   `nEqu × (nCells+nGhostCells)` 从 host 上传，计算后把 `dqdx/dqdy/dqdz` 完整下载
   回 MRField；随后 `LimField::CalcFaceValue` 只读 host `qf1/qf2`、gradient、
   limiter 和几何。数学上所需数据已经齐全，布局也是 equation-major，但现有接口没有
   device pointer/view，也没有 reconstruction → flux 的 device chaining seam。因此
   仅替换 `CalcFaceValue` 内层循环不能消除本轮 D2H。

2. **qf1/qf2 目前必然发生下一次 H2D。** `NsLimField::Init` 每次宏步为
   `nEqu × nFaces` 新建 host MRField；`GetQlQr`/`CalcFaceValue`/`BcQlQrFix` 在 host
   完成后，`UNsInvFlux::CalcAndAddInvFluxHipBatch` 通过
   `primitiveLeftComponents`/`primitiveRightComponents` 交给
   `HipFluxBackend::CalcAndAddPrimitiveFaceFlux`，backend 再逐方程 H2D。现有 fused
   API 只接受 host pointers，不能接收 reconstruction 输出的 device buffers。

3. **边界/ghost/ownership 语义已明确。** `UCom::Init` 定义
   `nTCell = nCells + nBFaces`；HIP gradient contract 强制
   `nGhostCells == nBoundaryFaces`。boundary faces 位于 `[0,nBFaces)`，每个 boundary
   face 的 `rightCell` 是对应 ghost；gradient kernel 先按 internal cell 归一化，再把
   left gradient 复制到 ghost。`GetQlQr` 先用 `q[left/right]` 初始化全部 face；
   `CalcFaceValue` 对两侧使用 face-center 到 cell-center 的位移和 limiter；随后
   `NsLimField::BcQlQrFix` 对前 `nBFaces` 逐面处理：`INTERFACE`/`PERIODIC` 保留重构值，
   其他边界退回 `q` 左右平均，`SOLID_SURFACE` 再以 face-indexed `bc_q` 覆盖两侧。
   residual ownership 是 left `-flux`，仅非 boundary 且 valid right cell 才加 right
   `+flux`。主 HIP path 当前不传显式 `boundaryMask`，所以仍依赖 boundary-first；
   trace metadata 也按 `face < nBFaces` 判定，不能在 reconstruction slice 中悄悄改变排序。

4. **应迁入 `Ns3DDeviceState` 的第一批持久 buffer。** state key 必须继续覆盖
   solver/zone/grid/backend，并在 restart/invalidate 时整体失效。受限
   limiter-off/inviscid slice 需要：cell `q`（含 ghost）、三方向 gradient、
   `qf1/qf2`、face geometry（face center/normal/area/mesh velocity）、cell geometry
   （center/volume）、left/right connectivity、显式 boundary mask、cell residual；只有
   FullTrace 才需要 device face flux D2H。`limiter` 在该 slice 可由 device 常量 `1`
   代替，`bc_q` 仅在存在 `SOLID_SURFACE` 时上传。RK、viscous/turbulence、halo/MPI
   不应混入第一刀。当前 `Ns3DDeviceState` 里的 vector 成员是 host scratch，HIP
   `DeviceBuffer` 仍由 process-wide shared backend 持有，尚未满足唯一 ownership。

5. **决策。** 先不实现孤立 reconstruction kernel。下一步应先建立 backend-neutral 的
   cell/face reconstruction view 和 state-owned buffer contract，再让 HIP backend 提供
   “gradient device buffer → qf1/qf2 device buffer → fused primitive flux/residual”
   的连续调用；CPU legacy/batch 继续作为 oracle。若实现阶段仍必须把 gradient 下载到
   MRField、再把 qf 上传，最多算架构接缝验证，不计入性能优化，也不启动 P2.7 多作业
   性能声明。


## 本轮 gradient device migration（2026-09-21）

- 新增 `CellGradientView` 与显式 opt-in `ONEFLOW_ENABLE_UNS_HIP_GRADIENT=1`；生产路径保留 CPU Green–Gauss oracle，不满足单 zone、finest grid、5 方程、Lax-Friedrichs、limiter off、inviscid 条件时 fail-fast。
- HIP backend 持久持有 gradient state、geometry 和 connectivity buffers；face-scatter 用 `atomicAdd`，internal-cell volume 归一化与 boundary ghost copy 分为线性 kernel。当前 q H2D 与 gradient D2H 仍每次发生，gradient device arrays 也尚未由 `Ns3DDeviceState` 唯一持有。
- 本地 CPU build 通过，CTest `243/243`；五算例 normal `1e-8` 与 fresh isolated strict `1e-15` 均 `5/5`，strict 最大绝对残差差 `1.1794086757568878e-17`；两个 F2 runner `bash -n` 和 `git diff --check` 通过。
- Kunshan target node：DTK 26.04 / `gfx906` / `dcu:1`，gradient smoke 对 CPU oracle 最大误差 `0`；GoogleTest `9/9`、hardware CTest `10/10`。
- 3-step、3-stage RK、m6、896256 faces / 294912 cells 的 36 条 trace 通过。legacy→CPU batch 最大绝对差 `2.2826185386293218e-13`；legacy→HIP gradient 最大绝对差 `3.4128255776977312e-13`，独立性能作业内重复 gate 为 `3.6970426720017713e-13`；metadata、finite、positive density/pressure、boundary semantics 与 conservation 全通过。
- fixed-CFL `0.01`、50-step、limiter off、inviscid：legacy 8 ranks、CPU batch 8 ranks、HIP gradient 1 rank 均 `exit_code=0`、`diagnostic_hits=0`、`PASS`，Slurm `COMPLETED/0:0`。
- 同 basis 单作业（`steps=3,warmup=1,repeats=3`）mean：legacy `14927.309397 ms`、CPU batch `15519.090773 ms`、HIP gradient `12866.935637 ms`，HIP/legacy `1.160129x`。HIP breakdown：`gradient=176.883 ms`、`reconstruction=933.296 ms`、`H2D=75.525 ms`、`D2H=14.485 ms`、`HIP kernel=10.163 ms`、`geometry_connectivity_H2D=5.718 ms`。`rk_update` 包含嵌套阶段，不能与子项相加。
- 结论：gradient migration 有明确单作业改善，下一优先级是 reconstruction。该证据不外推到 viscous/turbulence、MPI、多 zone/multigrid，也不外推 1D Euler 的 3.85–9.70x；未更新正式性能报告。

## 本轮新增证据（2026-09-19）

- **上游同步**：把 `upstream/master` `fa3f3b06`（PR #151–#158）merge 进 `dev`，解决 upstream `FieldPipeline` 重构与 dev `SimuContext` 透传的冲突；`master` = `origin/master` = `upstream/master` = `fa3f3b06`。
- **fresh CPU 五算例解除阻塞**：2026-09-19 在 `kshcnormal` 上按标准 `cpu-regression` 跑 `bc6d395b`，normal `1e-8` 5/5（最大绝对残差 `4.870783081880291e-11`）、strict `1e-15` 5/5（残差 `0.0`）；合并前基线 `730e9ae4` 复跑数值完全相同。recap 记录的 P0「被 continuation fixture/runner 阻塞」由此解除——它是 `f2b3d43c` 的 MPI rank-local state-sync 修复带来的，不是本次合并的功劳。
- **PR #160 的门禁是在 PR 分支自身上跑的**（不是拿 dev 的结果顶替）：T1 `cpu-regression` normal 5/5 + strict 5/5；T2 `dcu-single`（`kshdnormal` + `dcu:1`）GoogleTest 9/9 + CTest 9/9，revision `ef36b928`。
- **cherry-pick 的代价被实测**：拆 PR 时发现 WENO5 单独 cherry-pick 到 upstream 后 **HIP 编译失败**（`hip/hip_runtime.h` 被包在 `namespace oneflow_1d` 内），且依赖 Phase 3 的 `DeviceBuffer` 重构；`hardware;hip;dcu` 标签由 dev 独有的 `cmake/OneFLOWEulerContract.cmake` 提供，缺它则标准 runner 筛不到测试。结论：**必须对 PR 分支本身跑门禁**，不能假设 dev 绿就等于 PR 绿。
- **文档一致性**：`AGENTS.md` 补 fork 无关的分支模型；`CLAUDE.md` 删除；两份 DCU 交接文档的「尚未推送」表述已按事实校正。

## 本轮性能优化实测（2026-09-19）

- 在当前工作树上增加 HIP 5 方程 residual face-scatter kernel：由每个 cell 扫描全部 face 改为每个 face 向 owner cells 做 `atomicAdd`，理论复杂度由 `O(nCells*nFaces)` 降为 `O(nFaces)`。
- 尝试将 HIP flux buffer 提升为 backend 成员；发现 backend 若以 static 形式跨 runtime 生命周期保存，会在 runtime finalize 后析构触发 `OneFLOW accelerator runtime is not initialized`，已撤掉 static backend。当前优化只保留安全的 face-scatter 路径；真正跨 RK stage 的 buffer 复用尚未完成。
- 昆山 root HIP 构建、smoke、GoogleTest `9/9`、HIP CTest `10/10` 通过。
- 3D m6，`896256` faces、5 方程、3-stage RK、3 steps：完整 trace/物理语义 verifier 通过。HIP 对 CPU batch 的 invflux 最大绝对误差约 `1.8e-15`，residual 约 `2.7e-15`，state 约 `1.6e-15`；metadata 完全一致，finite/positive density-pressure/conservation 均通过。
- 性能 basis：同一输入、`steps=3`、`warmup=1`、`repeats=3`，端到端 wall-clock，8-rank CPU legacy/CPU batch 对比 1 DCU HIP batch：
  - legacy CPU：`23958.77 ms`
  - CPU batch：`24656.08 ms`，相对 legacy `0.9717x`（慢约 `2.91%`）
  - HIP/DCU batch：`24323.63 ms`，相对 legacy `0.9850x`（慢约 `1.52%`）
- 结论：face-scatter 没有带来应用级加速；当前主要瓶颈不在单一 flux/residual kernel。主 solver 当前 `UNsInvFlux` 路径仍主要是 host pack、H2D、HIP flux、D2H、host residual/state task 的串联，`FluxBackend::AddFaceFlux` 优化不能代表完整生产路径已经 device-resident。

## 本轮实现状态（2026-09-20）

- `master`、`origin/master`、`upstream/master` 已统一为 `063c0a12`；该主线已合入 `dev`（merge commit `0a1666eb`）。
- 主线合并后发现的 `SimuContext` 重复 accelerator state 定义已在 `6d6547c5` 清理。
- `UNsInvFlux::CalcFlux` 在 HIP batch 模式下走 `CalcAndAddPrimitiveFaceFlux`：primitive face state、HIP inviscid flux 和 residual face-scatter 在一次 device 调用中完成；NoTrace 时不回传 host flux，FullTrace/stage trace 才请求诊断性 D2H。
- 本地 CPU-only 根工程已用 OpenMPI 编译成功；CMake/CTest 4.2.0 下 `242/242` 测试通过。该结果不替代 Kunshan DCU 目标节点验证。
- 本轮已加入 `Ns3DDeviceState`：按 solver/zone/grid/backend key 注册，持有 conserved/old/residual/RK/gradient/limiter/reconstruction/face/geometry/connectivity/boundary/halo/viscous host scratch；HIP backend 的 shared 生命周期与 geometry/connectivity buffer cache 也已接入。该 vertical slice 尚未让全部 device arrays 由 state 唯一持有，gradient/limiter/reconstruction、viscous/turbulence、RK/state update 和 MPI/interface 仍在 host。
- 已加入 `ONEFLOW_STAGE_BREAKDOWN_FILE` opt-in 分段计时；`ci/kunshan/f2-main-solver-stage.slurm` 改为 8 MPI slots、trace timeout 可参数化，正式汇总只读取 `repeat-*`。本地 CPU-only 根工程重编译成功，CTest `242/242` 通过。
- Kunshan inviscid-only v4：DTK 26.04 / `gfx906` / `dcu:1`，root HIP smoke 通过、GoogleTest `9/9`、hardware CTest `10/10`；3-step、3-stage RK、36 条 trace 完整通过。HIP 对 legacy 的 pair 最大绝对差 `2.8554936193359026e-13`、最大 scaled error `1.7041923427996153e-13`；metadata 完全一致，finite/positive density-pressure/conservation 全通过。
- 同 basis：`test/m6wingroe_sa`，896256 faces / 294912 cells，`steps=3,warmup=1,repeats=3`，legacy/CPU batch 为 8 MPI ranks，HIP batch 为 1 rank，资源为 8 tasks × 1 CPU、27G、`dcu:1`，inviscid-only、limiter off、CFL `0.01`。raw mean 为 legacy `15058.725461 ms`、CPU batch `15724.756053 ms`（`0.957644x`）、HIP batch `13869.312341 ms`（`1.085759x`）。这是单次作业结果，不据此修改已发布性能报告。
- `breakdown.tsv` 共汇总 51 个正式 repeat/rank 文件、0 条 warmup 行、所有数值 finite。对每个 CPU repeat 先取 8 ranks 的 stage 最大值再对 3 repeats 求均值：HIP `rk_update=11044.900 ms`（包含嵌套阶段，不能与子项相加）、`initialization=2660.130 ms`、`gradient=1046.610 ms`、`reconstruction=921.434 ms`、`H2D=95.122 ms`、`D2H=14.669 ms`、`HIP kernel=11.463 ms`、`geometry_connectivity_H2D=8.552 ms`。因此下一条 inviscid-only 迁移路线选择 gradient/reconstruction；本轮禁用了 viscous/turbulence，不能据此给完整 NS 排序。
- fresh CPU v5：`kshcnormal` 16 CPU / 54G，normal `1e-8` 5/5（最大绝对残差 `4.970574442764598e-10`），strict `1e-15` 5/5（最大绝对残差 `1.1072414686508214e-17`）；scheduler `COMPLETED`、workload exit `0:0`。
- fresh 50-step stability v5b：legacy → CPU batch → HIP batch 顺序运行；三路均在第 13 步出现同一负压警告并在第 20 步失稳。legacy `ress=-nan` / exit `1`，CPU batch `non-finite Euler primitive state` / exit `1`，HIP `ress=nan` 后 teardown signal 11 / exit `139`。结论仍是共同物理失稳，不能归因于 HIP；HIP 失败路径另有析构问题。

## 上一轮证据（2026-09-16，保留）

> 以下是 2026-09-16 的快照。其中「fresh CPU 五 case 被 continuation fixture 阻塞」
> 已于 2026-09-19 解除（见上节）；其余结论（50-step 共同发散、无有意义加速）仍然成立。

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

### 提 PR 时的三个坑

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
3. **PR 分支必须独立跑门禁。** dev 绿 **不等于** PR 分支绿。2026-09-19 实测：
   单独 cherry-pick WENO5 到 `upstream/master` 后 HIP 编译失败（`hip/hip_runtime.h`
   被包在 `namespace oneflow_1d` 内），必须连同 Phase 3 与 `eee02bd6` 一起带；
   而 dev 的 `tests/euler/CMakeLists.txt` 还会引用未导入阶段的
   `EulerInvFluxCapability`。所以每个 topic 分支都要在昆山重跑 T1 `cpu-regression`
   +（涉及 DCU/HIP 时）T2 `dcu-single`，并把结果写进 PR 描述。

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

读完 **开头的「交接摘要」** + §1（现状）+ §2（待办）后选任务开工；**收工前更新本文档**。本文档只存在
于 `dev` 分支（本地 + fork），master/upstream 上都没有。

## 0. 怎么用这份文档

- **开工前**：读 §1 当前状态 + §2 待办，按优先级选一件。
- **收工前**：更新 §1（分支/PR/CI 状态）、勾选或新增 §2、把完成的移到 §5。
- **交接给他人/新会话**：对方只需读本文档 + 上表中两个入口。
- 本文档是活文档，不写日期后缀；历史事实放 §5。

## 1. 当前状态快照

| 项目 | 状态 |
|---|---|
| 主分支 | `master` = `origin/master` = `upstream/master` = `063c0a12`（三端 0/0；已合入 dev） |
| 进行中的 PR | **#159**（`pr/agents-branch-model`）：`AGENTS.md` 的 fork 无关分支模型，1 文件 +23/−0。**#160**（`pr/euler-weno5-unified`）：accelerator substrate + CPU vertical slice + 1D Euler port 的 WENO5/HIP contract，64 文件 +4771/−272。两者均 OPEN / MERGEABLE、CI 4/4 绿，都基于 `upstream/master` 分叉且不含 fork-only 文档 |
| 分支 | 本地 `dev` 将在本轮 fork-only 文档提交后推送到 `origin/dev`；共享 gradient 功能提交与本文档提交保持分离，未创建 upstream PR，也未 rebase。 |
| 昆山工作区 | 已规范化：`<workspace>/` 下 `src/`、`deps/`、`builds/`、`runs/<date>/<suite>/`、`archive/`；集群侧 README 记录具体路径 |
| 昆山作业脚本 | 四个标准套件脚本已更新到新工作区路径 |
| 智能体入口 | 仓库 `AGENTS.md`（含文档地图、分支模型与工作规则）；`CLAUDE.md` 已于 2026-09-19 删除；技能仓库 `oneflow-dev`（已安装到本地 skills 目录） |
| 测量口径 | 已确立：`lifecycle_*_ms` 为 repeats 总和，异口径不可比；历史 13.10× 勘误已修正为 25.55× |
| 当前进度 | F 阶段的 host-staged HIP 主路径已包含 primitive face state → HIP inviscid flux → residual face-scatter，并新增首个 HIP Green–Gauss gradient vertical slice。3-step accuracy、fixed-CFL 50-step stability 与同 basis timing 已通过；reconstruction、RK/state update、viscous/turbulence、MPI/interface 仍主要在 host。 |
| 最新验证 | 2026-09-21：本地 CPU build + CTest `243/243`，五算例 normal/strict 各 `5/5`；Kunshan DTK 26.04 / `gfx906` / `dcu:1` root HIP smoke（gradient oracle 误差 0）、GoogleTest `9/9`、hardware CTest `10/10`；m6 3-step 36 条 trace PASS；fixed-CFL 50-step 三路 PASS；同 basis mean 为 legacy `14927.309397 ms`、CPU batch `15519.090773 ms`、HIP gradient `12866.935637 ms`（`1.160129x`）。单作业结果未写入正式性能报告。 |

**能力边界（不要越界声明）**：一维 Euler 的 CPU/HIP 后端与单节点 MPI 已实测；CUDA、Kokkos、跨节点 MPI 和完整 Navier–Stokes 主线均未验证。`codes/accel` 的 accelerator substrate 已完成 Phase 1-3；3D 主 solver 当前只在单 zone、finest grid、5 方程、Lax-Friedrichs、limiter off、inviscid 条件下验证 HIP inviscid flux/residual 与 opt-in Green–Gauss gradient。q H2D、gradient D2H、reconstruction、RK/state update、viscous/turbulence 和 MPI/interface 仍依赖 host，不能称为完整 stateful/device-resident solver。

**GPU 融入路线图 (2026-09-13 启动)**：

| 阶段 | 内容 | 阶段含义（中文） | 状态 |
|------|------|------------------|------|
| Phase 1 | FluxBackend 扩展为 Euler 多方程 Rusanov（CPU+HIP kernel） | 先让批量通量 backend 能处理标量、3 方程 Euler 和 5 方程 NS 数据。 | ✅ `11b98029` |
| Phase 2 | 验证桥：FluxBackend vs port EulerBackend 数值一致性 | 用独立桥接测试证明新 backend 与已有 CPU oracle 的数值结果一致。 | ✅ `c4764c48`（机器精度 2.22e-16） |
| Phase 3 | HipEulerBackend 接入 AccelBackend + DeviceBuffer 统一设备管理 | 统一 accelerator runtime 和设备内存管理，避免 backend 各自维护重复资源。 | ✅ `983641c8` |
| Phase 4 | 主求解器 UNsInvFlux::CalcInvFlux 批量 GPU 化 | 把生产主 solver 的逐面通量循环逐步迁移为受控 HIP vertical slices。 | 🟨 3D m6 accuracy/fixed-CFL stability 已通过；gradient slice 完成，reconstruction 与完整 device ownership 待推进 |

**融合后的唯一执行主线**：架构 contract 解决生命周期/所有权，FluxBackend 解决批量通量计算；两者在主 solver CPU adapter 汇合，再复用到 HIP/DCU。

| 融合层 | 交付物 | 阶段含义（中文） | 状态 |
|---|---|---|---|
| A. 运行基线 | master 与 origin/upstream 同步 | 先固定共同代码基线，确保两条开发线从同一个上游版本继续。 | ✅ |
| B. standalone | 1D Euler CPU/HIP lifecycle、FullTrace/NoTrace、MPI 实验 | 用最小可控算例验证状态生命周期、CPU/HIP 后端和 MPI 基础能力。 | ✅ |
| C. accel substrate | AccelRuntime、AccelBackend、FluxBackend、DeviceBuffer | 建立与具体 solver 解耦的运行时、设备内存和批量 kernel 基础设施。 | ✅ Phase 1-3 |
| D. domain contract | EulerDomain views、StateRegistry；通用 views 的布局/几何/能力元数据已补齐 | 明确 solver 与 accelerator 之间的数据、所有权和生命周期契约。 | ✅ E1、E2 与生命周期 service、主 solver INIT/restart hook 已完成 |
| E. CPU vertical slice | INIT_FLOWFIELD、CPU adapter、RungeKutta、CPU oracle | CPU 主 solver 初始化、通量、时间推进、逐面 trace 与物理门禁已闭环。 | ✅ E1–E6 |
| F. DCU vertical slice | root 生产 HIP 编译、smoke、contract、3D m6 accuracy/fixed-CFL stability 与首个 device-gradient slice 已闭环；完整 device ownership、NS/MPI 和连续多作业性能仍未完成 | 当前证明的是受限 capability 下的 host-staged inviscid vertical slice，不是完整 stateful/device-resident 3D solver。 | 🟨 |
| G. MPI/性能 | host-staged halo、GPU-aware probe、reduction、性能 | 最后处理跨 rank 数据交换、设备归约和端到端规模化性能。 | ⬜ |

**整合约束：** `FluxBackend` 接收 equation-major conserved face state；`UNsInvFlux` 提供 reconstructed primitive state，adapter 负责转换，backend 负责面面积；face connectivity 仍由主 solver 的 `AddF2CField` 处理。CPU batch 入口必须以旧 CPU 逐面路径为 oracle。

**port / accel / NS 三层关系（统一后）**：

| 层 | 角色 | GPU 状态 |
|----|------|----------|
| **port** (`ports/kunshan/`) | 最小实验闭环：1D Euler, persistent state, FullTrace, 合约测试 | ✅ 完整 |
| **accel** (`codes/accel/`) | 生产基础设施：AccelRuntime, FluxBackend, DeviceBuffer | ✅ 完整（标量+Euler） |
| **NS** (`codes/ns/` + `codes/uns/`) | 主求解器：5 方程；当前 Lax-Friedrichs HIP batch 已包含 inviscid flux/residual 与 opt-in Green–Gauss gradient | 🟨 单 zone/finest-grid/inviscid/limiter-off 已验证；reconstruction、RK/state、viscous/turbulence、MPI/interface 仍主要在 host |

**OpenAI NS 解决公告 (2026-09-08) 参考**：
- 数学证明（奇点存在性），非数值求解器，与 OneFLOW 直接技术关联有限
- 方法论可借鉴：Euler 先做热身 → NS；形式化验证 1/6 工时（对应 Phase 2 桥接测试）
- 行业信号：AI+CFD 交叉领域在加速，GPU 求解器基础设施具有战略价值

## 2. 待办事项

### P0 — upstream PR 跟进（2026-09-19 起）

- [ ] **PR #159**（`pr/agents-branch-model`）：跟进 review；如需改动，在该 topic 分支上改并重跑 CI。
- [ ] **PR #160**（`pr/euler-weno5-unified`）：跟进 review；**任何改动都必须在该 PR 分支上重跑 T1 `cpu-regression` + T2 `dcu-single`**——本轮实测过「dev 绿 ≠ PR 分支绿」（单独 cherry-pick WENO5 会导致 HIP 编译失败），不要拿 dev 的结果顶替。
- [ ] 两个 PR 合并或关闭后删除对应的 topic 分支（本地 + origin）。

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
  - [x] F1：主 solver HIP backend registration。说明：在不改变 CPU 默认路径的前提下，将 HIP backend 与 capability/fail-fast guard 接入生产 solver；`HipFluxBackend` 现为 runtime 生命周期内复用的 shared 执行对象，持有持久 device buffers 与 geometry/connectivity cache，但仍不是按 solver/zone/grid 注册的 `Ns3DDeviceState`，也不是全部 device arrays 的唯一 owner。
    - [x] F1.1：`d5005ad6` 完成统一 `FluxBackend&` 注入与 CPU/HIP 共用 pack；`ONEFLOW_ENABLE_HIP_TESTS=ON` 联动生产 HIP backend。
    - [x] F1.2：`ca14118c` 将 `UNsInvFlux` HIP path 限定为受支持 solver、5 方程、Lax-Friedrichs、单 local zone、finest/single grid，并逐项返回稳定拒绝原因。
    - [x] F1.3：默认/CPU 路径不变；显式请求 HIP 而 build/runtime/backend/capability 不满足时 fail-fast，不静默回退。
  - [x] F2：主 solver CPU/HIP numerical gate。说明：先用小 case 复现 CPU oracle，再逐步扩大到 3D m6 case。
    - [x] F2.1：adapter one-call CPU/HIP oracle；257 faces × 5 equations，覆盖 3D normals、ALE mesh-normal velocity、face area 与非 boundary-first 显式 mask；flux 最大差 `6.661e-16`，residual 最大差 `1.110e-15`。
    - [x] F2.2：HIP 小 case 1-step LU-SGS 与 legacy CPU、CPU batch 对比，检查 `qf1`、`qf2`、face flux、residual、state trace。
    - [x] F2.3：小 case 检查 finite、positive density/pressure、boundary semantics、conservation、ALE 与 face-area ownership。
    - [x] F2.4：扩大到 3D m6 case，并完成长步稳定性门禁。
      - [x] F2.4a：3D m6 896256 faces、Lax-Friedrichs、3-stage RK 1-step；HIP 对 legacy 最大差 `1.706e-13`，finite 且 density/pressure 为正。
      - [x] F2.4a-3：2026-09-20 inviscid-only v4 的 3-step 三路 trace 完整 verifier 通过；HIP invflux/residual 最大绝对差均不超过 `3.109e-15`、state 不超过 `2.887e-15`，三路 pair 最大绝对差 `2.8554936193359026e-13`，metadata 完全一致，finite/positive/conservation 全通过。
      - [x] F2.4b：fixed-CFL 50-step 稳定性已闭环。runner 默认令 `cflst=cfled=0.01`，并把有效参数写入 `configuration.tsv`；`fixed_inviscid`、`fixed_limiter`、`fixed_lowcfl`、`fixed_viscous` 四组的 legacy、CPU batch、HIP batch 均 `exit_code=0`、无非物理诊断，不能再把旧 ramp 配置下的第 14 步负压当作 HIP 发散证据。
      - [x] F2.4c：inviscid-only 的 3-step accuracy/breakdown/timing 与 fixed-CFL 50-step stability 已完成；2026-09-21 又完成 opt-in HIP Green–Gauss gradient slice。当前 gradient 每次仍做 q H2D 与 dq D2H，reconstruction、RK/state update、viscous/turbulence 和 MPI/interface 仍主要在 host，不能把当前 3D timing 与 1D Euler stateful 报告比较。
  - [x] F3：主 solver DCU target-node evidence。说明：记录 DTK、gfx906、visible device、资源 tuple 和 workload exit code。
    - [x] F3.1：标准 root HIP runner 已沉淀；支持 trace step 参数化、精度门禁后 benchmark、同 basis timing 与非空 HIP test 检查。
    - [x] F3.2：2026-09-20 v4 在 Kunshan DTK 26.04 / `gfx906` / `dcu:1` 完成 root HIP build、smoke、GoogleTest `9/9`、hardware CTest `10/10`、3-step trace gate、正式 repeat breakdown 和同 basis timing；2026-09-21 fixed-CFL 四组 stability 作业的 scheduler/workload 均为 `COMPLETED/0:0`，legacy/CPU batch/HIP batch 全部 50-step PASS。raw mean 为 legacy CPU `15058.725461 ms`、CPU batch `15724.756053 ms`、HIP batch `13869.312341 ms`，raw ratio `1.085759x`；该 timing 仍是 host-staged 单次作业结果，不作为正式稳定加速结论，也未更新性能报告。
    - [x] F3.3：fresh CPU 五 case 门禁于 2026-09-20 在 v5 快照上复验：normal `1e-8` 5/5（最大绝对残差 `4.970574442764598e-10`）、strict `1e-15` 5/5（最大 `1.1072414686508214e-17`）；scheduler `COMPLETED`、workload exit `0:0`。
    - [x] F3.4：2026-09-21 gradient slice 在同一 Kunshan target-node 口径完成 root HIP build/smoke/contract、3-step 36-record accuracy、fixed-CFL 50-step stability 与同 basis timing；三路 workload 与 scheduler 均成功。HIP/legacy 单作业 mean `1.160129x`，但尚无连续多作业重复，因此不更新正式性能报告。
- [ ] 昆山回归 eric 的完整 `task/database/register/adt` 测试套件。

### P2 — 后续技术工作

说明：先完成可观测性和 host-staged 路径优化，再做设备归约、WENO5 DCU 验证和性能优化。每项性能改动都必须在同一输入、同一 `steps/warmup/repeats` basis 下比较，并保留 accuracy gate 结果。

- [x] **P2.0：主 solver stage 分段计时**
  - [x] `StageProfiler` 已在初始化、gradient、limiter、reconstruction、boundary reconstruction、host pack、geometry/connectivity H2D、H2D、HIP kernel、D2H、viscous/turbulence、RK update、residual/state update、MPI/interface 和 output 边界提供 opt-in 计时。
  - [x] `ci/kunshan/f2-main-solver-stage.slurm` 输出 machine-readable `breakdown.tsv`；正式汇总只读取 `repeat-*`，与 `steps=3,warmup=1,repeats=3` 的 timing basis 分开且不混入 warmup。
  - [x] 2026-09-20 Kunshan target-node 已完成 inviscid-only breakdown：51 个正式 repeat/rank 文件、0 条 warmup 行、所有数值 finite。HIP 的 per-repeat mean 为 `gradient=1046.610 ms`、`reconstruction=921.434 ms`，显著高于 `H2D+D2H+HIP kernel+geometry H2D` 约 `129.805 ms`；下一迁移路线据此选择 gradient/reconstruction。`rk_update` 包含嵌套阶段，不能与子类别相加；viscous/turbulence 本轮禁用，完整 NS 仍需另测。

- [ ] **P2.1：把 HIP backend/state 绑定到宏步生命周期**
  - [x] `HipFluxBackend::Shared()` 已让同一 runtime 内的 flux/gradient 调用复用长期 backend 与 device buffers，不再在每次 `CalcInvFlux` 创建临时 backend。
  - [x] `SimuContext::TeardownEnvironment()` 在 accelerator runtime finalize 前先 clear registry state、再 `ReleaseShared()`，旧 teardown signal 11 已消除。
  - [ ] 把 shared backend/device buffers 收敛到按 solver/zone/grid/backend key 管理的 `Ns3DDeviceState`，补 teardown/reinitialize contract，形成唯一且可失效的 device ownership。
  - [x] 连续多个 RK stage、3-step accuracy 与 fixed-CFL 50-step 重复宏步均无 device allocation/lifecycle 错误。

- [x] **P2.2：缓存不变的 device geometry/connectivity**
  - [x] flux/residual 路径缓存 `leftCell/rightCell`、`boundaryMask`、三维 normals、face area 与 mesh velocity；gradient 路径缓存 face/cell geometry、volume 与 connectivity。
  - [x] cache 以 grid/owner key、extent 和 host pointer identity 校验；identity 变化时重新上传，backend/device lifecycle 结束时随 shared backend 释放。
  - [x] 3-step accuracy/metadata gate 不变；gradient 单作业 breakdown 中 `geometry_connectivity_H2D=5.718 ms`，后续 ownership/invalidation 统一归入 P2.1。

- [ ] **P2.3：减少每个 RK stage 的 H2D/D2H**
  - [x] 第一刀：`13275297` 让 HIP batch 直接从 MRField 方程分量上传/回传，移除连续 face/residual host pack/unpack；单次作业 raw ratio 从 `1.034361x` 变为 `1.153677x`，但跨两次作业的 HIP mean 仅改善约 `1.995%`，需更多重复作业确认。
  - [ ] 评估 qf1/qf2 是否可直接写入长期 device state，避免 contiguous host pack 后完整 H2D。
  - [ ] 将 flux 与 residual update 合并为生产路径中的连续 device 操作，避免完整 `invflux` D2H 后再由 host 回写 residual。
  - [ ] 优先实现单 zone/finest grid/Lax-Friedrichs 五方程路径，其他 capability 保持 legacy fallback。
  - [ ] 验收：3-step accuracy gate 通过；D2H/H2D 总字节数和 stage breakdown 明显下降。

- [x] **P2.4：让 face-scatter 接入主 solver 生产路径（第一步）**
  - [x] 已确认 `UNsInvFlux::CalcFlux` 的 HIP batch 调用链，并让 `CalcAndAddPrimitiveFaceFlux` 在主 solver 中执行 fused face-scatter residual kernel；不再只优化未被 benchmark 使用的 `FluxBackend::AddFaceFlux` API。
  - [ ] 比较 atomic face scatter、cell adjacency/segmented reduction 两种 residual 回写方案，记录 atomic contention 风险。
  - [ ] 验收：主 solver HIP batch 的 residual 回写确实走 device kernel，并通过 conservation、state trace 和 50-step stability gate。

- [x] **P2.5：HIP Green–Gauss gradient vertical slice**
  - [x] 新增 equation-major cell/ghost gradient contract、显式 opt-in 与 capability fail-fast；保留 CPU Green–Gauss oracle。
  - [x] geometry/connectivity cache、device accumulation/normalize/boundary-copy kernel 与 synthetic CPU-oracle smoke 已完成。
  - [x] 本地 build/CTest、Kunshan target-node contract、3-step accuracy 和 fixed-CFL 50-step stability 全通过。
  - [x] 同 basis 单作业显示 HIP `gradient` 由约 `1046.610 ms` 降为 `176.883 ms`，端到端 HIP/legacy 为 `1.160129x`。
  - 边界：当前仍每次 q H2D、gradient D2H，且 device buffers 尚未由 `Ns3DDeviceState` 唯一持有。

- [ ] **P2.6：HIP reconstruction vertical slice**
  - [x] 完成 limiter-off reconstruction 的 MRField/device contract、face ownership、boundary reconstruction 顺序与 trace oracle 只读盘点；当前结论是不先写仍需 gradient D2H + qf H2D 的孤立 kernel。
  - [ ] 先建立 backend-neutral cell/face reconstruction view，并把首批 q/gradient/qf1/qf2/geometry/connectivity/residual buffers 纳入按 solver/zone/grid/backend key 管理的 state-owned contract。
  - [ ] 只覆盖单 zone、finest grid、5 方程、Lax-Friedrichs、limiter off、inviscid；不同时迁移 limiter、RK 或 viscous/turbulence。
  - [ ] 按 3-step accuracy → fixed-CFL 50-step stability → 同 basis timing 顺序验收。

- [ ] **P2.7：重新测量端到端性能**
  - [x] 已固定 `steps=3,warmup=1,repeats=3`、m6 输入、CPU 8 ranks、HIP 1 rank 和同一资源 tuple，完成 gradient slice 的首个正式单作业测量。
  - [x] 首个阶段目标 `HIP batch < legacy CPU` 已在该单作业达到，但不提前承诺稳定倍数。
  - [ ] 只有连续多次测量方向一致且 accuracy/50-step/MPI correctness 全通过，才更新性能报告。


- [ ] **GPU reduction**（优化计划阶段 D 唯一剩余项）
  - 内容：checksum、最大误差、有限性/正状态检查放到设备端归约，只回传标量。
  - 参考：`doc/plans/oneflow-euler-optimization-plan.md` 阶段 C/D。
  - 验收：Kunshan 四规模 correctness 不变；D2H 占比进一步下降；性能复测。

- [ ] **WENO5 数值内核的 DCU 四规模验证**（单卡 contract 已完成）
  - [x] `nx=32` contract：WENO5 HIP 与 CPU oracle 对齐。
  - [ ] `nx=65536/262144/1048576/4194304` correctness 与性能复测。

### P3 — 维护与清理

说明：清理临时分支和工作区，并把验证后的协作流程沉淀回技能与文档。


- [ ] 本地分支清理：`pr/agents-branch-model`、`pr/euler-weno5-unified` 在对应 PR 合并/关闭后删除（本地 + origin）；`fix/contract-test-cmake-path` 已于 2026-09-19 删除（内容早经 PR #149 合入 master）。dev 上已无其他遗留分支。
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
6. **提 PR 的边界**：dev 上同时存在**共享文件**（`AGENTS.md`、`ci/`、`cmake/`、
   `ports/`、`tests/`、`codes/`——上游也有）与 **fork-only 文档**
   （`doc/plans/oneflow-development-todo.md`、`doc/handoff/` 的工作记录——永不进上游）。
   两者必须**分开 commit**，否则将来无法单独 cherry-pick 共享文件。动手前用
   `git cat-file -e upstream/master:<path>` 判断归属。详见 §协作约定。

## 4. 每轮任务的收尾流程（Definition of Done）

1. 本地构建 + 相关测试套件通过；数值改动附五算例结果。
2. 推送后等待 CI 绿灯；红灯不算完成（原 #148 的教训）。
3. 更新本文档：§1 状态、§2 勾选/新增、完成的移入 §5。
4. 只有满足对应报告的发布准入条件（同 basis、足够重复、accuracy/stability/MPI 门禁完整）的新测量才更新报告；更新时 Markdown 与 HTML 必须同步。探索性单作业数据只记入本文档或外部运行证据。
5. 如发现已发布数据的错误：修正维护中的报告并加勘误说明（参考 4-DCU 基线的处理）。

## 5. 已完成（按时间倒序）

| 日期 | 事项 | 证据 |
|---|---|---|
| 2026-09-21 | 完成首个 3D 主 solver HIP Green–Gauss gradient vertical slice；共享代码与 fork-only TODO 分开提交 | 本地 CPU build + CTest `243/243`、五算例 normal/strict 各 `5/5`；Kunshan DTK 26.04 / `gfx906` / `dcu:1` gradient smoke 误差 0、GoogleTest `9/9`、hardware CTest `10/10`、3-step 36-record accuracy PASS、fixed-CFL 50-step 三路 PASS、同 basis HIP/legacy `1.160129x`；当前仍 host-staged，不是完整 stateful/device-resident |
| 2026-09-21 | 定位并修正 stability runner 的 CFL ramp 配置错误：旧 runner 只改 `cflst`、保留 `cfled=10`，导致第 14 步实际 CFL 约 1.41；新增 fixed-CFL、显式 limiter 和 configuration.tsv | Kunshan DTK 26.04 / `gfx906` / `dcu:1`：`fixed_inviscid`、`fixed_limiter`、`fixed_lowcfl`、`fixed_viscous` 四组均为 legacy/CPU batch/HIP batch `exit_code=0`、`diagnostic_hits=0`、`PASS`，Slurm `COMPLETED/0:0`；当前 3D 仍是 host-staged，不是完整 GPU-resident |
| 2026-09-20 | 3D persistent-state / stage-breakdown vertical slice：新增 `Ns3DDeviceState` 注册与 host scratch ownership、HIP geometry/connectivity cache、opt-in `StageProfiler`、正式 repeat breakdown 汇总、8-rank Slurm slot 修正、可参数化 trace timeout 和独立 50-step stability runner | 本地 CPU build + CTest `242/242`；Kunshan v4 root HIP smoke、GoogleTest `9/9`、hardware CTest `10/10`、3-step accuracy、breakdown 和 timing 全通过；v5 CPU normal/strict 各 5/5；v5b 50-step 三路均于第 20 步共同失稳，HIP 错误路径另有 teardown signal 11。当前 3D 仍是 host-staged，不是完整 GPU-resident |
| 2026-09-19 | 开出 upstream **PR #160**（`pr/euler-weno5-unified`）：accelerator substrate（Phase 1–3 + E1–E6 CPU vertical slice，含主 solver gated CPU batch 路径、mutable `SimuContext` 透传、MRField 生命周期绑定、RungeKutta capability guard）与 1D Euler port 的 WENO5 / AccelBackend 设备管理 / HIP contract 注册；**门禁在 PR 分支自身上跑**：T1 `cpu-regression` normal 5/5 + strict 5/5，T2 `dcu-single` GoogleTest 9/9 + CTest 9/9，CI 4/4 绿 | revision `ef36b928`；64 文件 +4771/−272；F 阶段（50-step 未收口、无端到端加速）刻意排除在外，并在 PR 描述中写明边界 |
| 2026-09-19 | 开出 upstream **PR #159**（`pr/agents-branch-model`）：`AGENTS.md` 增加 fork 无关的分支模型（baseline / working / topic 三类分支，`dev` 引用带存在性条件，不引用 fork-only 文件）；CI 4/4 绿 | revision `b4c041c6`；1 文件 +23/−0；共享文件单独成 PR，不与 dev 的 fork-only 文档混在一起 |
| 2026-09-19 | 拆 PR 时实测出三个必须记住的坑：① 单独 cherry-pick WENO5 到 upstream 后 **HIP 编译失败**（`hip/hip_runtime.h` 被包在 `namespace oneflow_1d` 内），必须同时带上 Phase 3 与 `eee02bd6`；② `hardware;hip;dcu` CTest 标签只存在于 dev 独有的 `cmake/OneFLOWEulerContract.cmake`，缺它标准 runner（`ctest -L hardware -R HIP`）筛不到测试；③ dev 的 `tests/euler/CMakeLists.txt` 引用了未导入阶段的 `EulerInvFluxCapability`。结论：**PR 分支必须独立跑门禁，不能用 dev 的结果顶替** | 已同步记入本文档 §协作约定；两次失败与修复过程见 PR #160 的提交 `67dc7fae`、`ef36b928` |
| 2026-09-19 | 文档一致性收口：`AGENTS.md` 补分支模型（并改为 fork 无关措辞）、删除 `CLAUDE.md`、校正两份 DCU 交接文档的「尚未推送」表述；共享文件与 fork-only 文档**分开提交**，便于将来增量 cherry-pick | commits `ac650a54`（AGENTS.md）、`c608de63`（删 CLAUDE.md）、`0e48c8e3`（fork-only 状态文档） |
| 2026-09-19 | 删除本地与 origin 的 `fix/contract-test-cmake-path`（内容早经 PR #149 合入 master）；`origin/gh-pages` 与 `upstream/gh-pages` 强制对齐 | 分支删除与 force-push 均已核对为 0/0 |
| 2026-09-19 | 昆山 `cpu-regression` 复验（标准 T1，`kshcnormal` 16 CPU）：`bc6d395b` 五算例 normal `1e-8` 5/5（最大绝对残差 `4.870783081880291e-11`）、strict `1e-15` 5/5（最大绝对残差 `0.0`）；合并前基线 `730e9ae4` 复跑数值完全相同，确认该合并无数值影响；recap 记录的 P0（fresh CPU 五 case 被 fixture 阻塞）由此解除 | `cpu-regression`；50-step 稳定性（P1）与真实加速（P2）仍未收口 |
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
