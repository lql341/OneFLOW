# 3D HIP 性能交接（2026-10-25：device-authoritative q 落地）

> 用法：把本文件整段贴给新的 agent 会话。规则以仓库 `AGENTS.md` 为准；
> 上一轮状态见 `doc/handoff/oneflow-3d-hip-handoff-20261024.md`。

## 0. 一句话现状

受限无粘 3D 路径（8 方程/LAX-Friedrichs/limiter-off/显式 RK）的 **q 已改为设备权威**：
状态更新不再回写 host，边界 ghost/墙面状态按需回拉，host-only 派生量按需重建。
同机 paired A/B（同二进制、8 zone/8 rank/1 DCU、1000 步、50 步 warmup）：
**38.26 → 31.64 ms/step（−6.62 ms，−17.3%，1.21×）**。门禁全绿（3 步 trace + resident
24/0/0、200 步跨后端 residual `2.33e-18`、aero `0.0`）。

## 1. 本轮改了什么（全部有门禁）

| 内容 | 位置 | 说明 |
|---|---|---|
| `ResidentStateContract` 增加「host 镜像是否有效」 | `codes/accel/include/ResidentStateContract.h` | `NoteDeviceFieldAdvance`（设备推进、host 变脏）/`NoteHostMirrorRefresh`/`NoteDeviceGhostWrite(mirrored)`/`NoteHostMirrorStale`；`HostMirrorCurrent()` 是唯一判据 |
| `ONEFLOW_HIP_DEVICE_AUTHORITATIVE_Q`（默认开） | `HipProductionPolicy.h` | steady + 显式 RK 且 backend=HIP 时启用；`startStrategy==2`（双时间）排除 |
| 状态更新不再 D2H | `codes/uns/src/UNsUpdate.cpp` | 每 stage 5 个方程的 q 下载消失；host `tempr` 循环同时跳过 |
| 边界 ghost/墙面状态按需回拉 | `codes/ns/src/NsSolverImp.cpp`、`HipFluxBackend.hip` | `bnd_ghost_copy` 只剩 bcQ（0.69 ms/step）；ghost q 改为按需 |
| 边界相邻 cell 的按需 q 镜像 | `HipFluxBackend::MirrorPrimitiveCellsToHost` + `NsMirrorHostBoundaryState`（`codes/ns/src/NsRhs.cpp`） | 气动力/压力系数每步只读边界相邻 owner cell；一次打包 kernel + 一次 D2H |
| 输出步整场回拉 + `gama/tempr` 重建 | `NsRefreshHostStateMirror`；`Restart::Dump`、`Visualization`、stage trace | 保存/可视化/门禁 trace 前把 owned+ghost 拉回，并用刷新后的 q 重建 host 派生场 |
| host interface 不再打包 q/tempr/gama | `codes/interface/src/InterfaceTaskReg.cpp` | 这些字段在设备路径上是 stale 的，pack/download 纯成本 |
| `UploadBoundaryState` 不再用旧 host 镜像覆盖设备 | `HipFluxBackend.hip` | 之前 owner generation 一变就整场 H2D；设备权威时那会用**旧**场覆盖设备 |
| 上传保护 | `RefreshResidentPrimitiveState` | mirror 脏时 `FullField` 上传直接抛错，避免静默回灌旧场 |

诊断/回退开关：`ONEFLOW_HIP_DEVICE_AUTHORITATIVE_Q=0` 回到旧的「每 stage 回写 host」；
`ONEFLOW_HIP_GHOST_Q_MIRROR=1` 只把 ghost q 的 D2H 加回来（bisect 用）。

## 2. 关键测量

### 2.1 paired A/B（同二进制、同节点、`runs/devqab-125498139`）

| 变体 | 1000 步 wall ms → ms/step |
|---|---|
| 旧行为（`..._Q=0`） | 37853.9 / 38927.8 / 38063.8 / 38196.2 → **38.26 ms/step** |
| 设备权威（默认） | 31356.4 / 31463.2 / 31913.7 / 31831.0 → **31.64 ms/step** |

paired delta = −6.62 ms/step（−17.3%）。**注意**：同一节点上旧行为本轮测得 38.3，
而 2026-10-24 记录的是 32.8；跨会话绝对值不可比，一律用 paired 差。

### 2.2 500 步分项（rank 0，ms/step）

`update_residuals 12.19`（含 `gradient 6.24`（含 `gradient_mirror 3.28`）、
`reconstruction 2.78`、flux kernel）、`interface_mpi 3.71`、`boundary_device 3.03`
（含 `bnd_resize 2.13`，仍是首个 `hipMalloc` 摊到 500 步的假象）、`load_residuals 2.52`、
`load_q 1.11`、`post_upload 1.04`、`swap_upload 1.03`、`calc_boundary 0.95`、
`bnd_ghost_copy 0.69`、`state_update 0.29`。

⇒ 剩余固定开销的主项已经换人：**gradient 与其 host 镜像、interface 每步 4 轮、
`load_q`/`load_residuals` 两个没人读的 host 整场拷贝、boundary+recon 的多次 launch**。

## 3. 下一轮（按优先级）

### P0.1 gradient host 镜像只在最后一个 stage 做（约 −2 ms/step）
`UNsInvFlux::CalcInvFace` 的 `hostMirrorReaderThisStep` 现在每个 stage 都为真
（因为 `Iteration::InnerOk()` 每步都为真，压力系数 dump 每步都跑）。
但 post-process 读到的必须是**最后一个 stage** 的梯度，所以只需在最后一 stage 镜像一次。

### P0.2 `LOAD_Q` / `LOAD_RESIDUALS` 在设备路径下跳过（约 −2–3 ms/step）
两者都是 host 整场拷贝（`q→dq`、`res` 取负），设备路径完全不读；但 restart 会写它们，
所以在 `nFieldSave`/输出步之前需要一次回拉（可复用 `NsRefreshHostStateMirror` 的思路）。

### P0.3 每步一次 halo 交换（约 −2.8 ms/step）
现在 4 轮：3 stage + 1 post。`interface_mpi 3.71 ms/step`。

### P0.4 stage kernel 融合（residual/recon/flux/lhs/update/boundary 合并）（约 −6 ms/step）
`boundary_device 3.03` + `reconstruction 2.78` + `gradient 6.24` 的 launch/同步点。

## 4. 这一轮踩到的坑（都已修，别再犯）

1. **设备权威不能从初始化就开始**：bound 状态第一次 `FullField` 上传时，host 的 ghost
   区是唯一完整来源；如果在初始化就把 ghost D2H 关掉，第一次上传会把初始化残留的 ghost
   带进设备，第一步就偏离 CPU。实现上用 `HipFluxBackend::HostMirrorStale(grid)` 作为
   是否跳过的条件（只有设备真的更新过才跳过）。
2. **设备写 ghost 但没回写 host 时，必须把镜像标脏**（`NoteDeviceGhostWrite(false)`）。
   否则 contract 诊断会显示 `mismatched_values ~ nGhost`（差值 1e-16 级，来自设备核与
   host 边界模型两条公式的末位差），并且之后任何一次 `FullField` 上传都会把旧 ghost
   灌回设备。
3. **`UploadBoundaryState` 是隐形回灌点**：它按 owner generation 判断是否需要整场 H2D。
   设备权威时 generation 每 stage 都前进，于是每个邻居、每个 stage 都用旧 host 场覆盖
   设备。现在改成「contract 说设备已是最新」就直接复用。
4. **resident bcQ 复用做过了，已撤回**：跳过 host bcQ 刷新后，只要 solver 侧与 backend
   侧对「能否复用设备墙面状态」的判断有任何一次不一致，就会上传**未初始化**的
   `hipBoundaryQ`，3 步内 `HIP primitive state update produced a non-physical state`。
   收益只有约 0.3–0.7 ms/step，不值得承担；host bcQ 的 D2H + 上传保留。
5. **排障技巧**：`Simulation::RunImpl` 的 `catch(...)` 会先 `MPI_Finalize` 再重抛，所以
   跨 rank 抛异常时 stderr 可能永远打不出来（其余 rank 卡在 MPI）。临时在 catch 里先
   打印 `e.what()` 再 finalize，能立刻定位；本次就是这么找到 non-physical 的。
6. 门禁 trace 的 resident contract 诊断现在会**先按需回拉**再从设备一侧比较，
   因为「host 镜像落后」是本轮的预期状态。

## 5. 门禁命令（昆山 worksapce）

```bash
# 3 步多区 trace + resident contract（全部 HIP stage 显式打开）
ONEFLOW_MZ_SOURCE_CASE=$ROOT/work/m6-8zone-cpu8/case \
ONEFLOW_MZ_CPU_BINARY=$ROOT/builds/root-instr/bin/OneFLOW \
ONEFLOW_MZ_HIP_BINARY=$ROOT/builds/root-instr/bin/OneFLOW \
ONEFLOW_MZ_RUN_ROOT=$ROOT/runs/<new> \
ONEFLOW_MZ_TRACE_VERIFIER=$SRC/ci/kunshan/f2-main-solver-stage-verify-stream.py \
ONEFLOW_MZ_RANK_VERIFIER=$SRC/ci/kunshan/verify-rank-utilization.py \
ONEFLOW_MZ_RANKS=8 ONEFLOW_MZ_STEPS=3 \
ONEFLOW_MZ_HIP_GRADIENT=1 ONEFLOW_MZ_HIP_RECONSTRUCTION=1 \
ONEFLOW_MZ_HIP_STATE_UPDATE=1 ONEFLOW_MZ_HIP_BOUNDARY=1 \
ONEFLOW_MZ_HIP_TIME_STEP=1 ONEFLOW_MZ_RESIDENT_CONTRACT=1 \
bash $SRC/ci/kunshan/f2-main-solver-multizone-trace.sh
# 200 步跨后端
ONEFLOW_XB_* ... bash $SRC/ci/kunshan/f2-main-solver-cross-backend-residual.sh
# CPU 五算例 normal/strict
python3 $SRC/test/test.py "mpirun -np 1" $BIN \
    $SRC/test/suites/cpu-serial.txt $SRC/test/baselines/residual-baseline.json <tol>
```

`/public/home/luql/.scnet-hpc/oneflow/scripts/{devq-gate,devqab,cpureg}.slurm` 是本轮
用过的作业脚本（集群侧，不入库）。
