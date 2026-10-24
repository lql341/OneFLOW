# 3D HIP 性能交接提示词（2026-10-24）

> 用法：把本文件整段贴给新的 agent 会话即可；它自带上下文、证据、决策和下一步。
> 分支：`dev`（本地 HEAD `b4ed31d3`，领先 `origin/dev` 1 个提交，未 push）。
> 规则仍以仓库 `AGENTS.md` 为准；本文只补当前状态。

## 0. 一句话现状

受限无粘 3D 路径（8 方程/LAX-Friedrichs/limiter-off/显式 RK）的**单卡 8:1 槽位**已经从
`3.89x` 做到 **`6.2–7.8x`**；单步时间 `59.4 → 32.8 ms`。再往上被**每步 22 ms 的固定
host 开销**卡住，下一步是 RK 移植（device 权威 + kernel 融合 + 每步一次交换），
目标 12–14 ms/步（14–16x）。多卡的正确评估还缺一个 ≥1M cell 的合法 fixture。

## 1. 环境（这几条踩过坑，直接照做）

| 项 | 值 |
|---|---|
| 集群 | 昆山 `kseshell`（OpenAPI region `11250`），`kshcnormal`=CPU，`kshdnormal`=DCU |
| MPI 运行库 | **必须显式设 PATH/LD_LIBRARY_PATH**：`scripts/mpienv.sh`（约定了 gcc16 / openmpi 5.0.11 / ucx 1.22.0）|
| 为什么 | 批处理里 `module load` 是空操作（`command -v mpirun` 为空、`ldd` 找不到 `libmpi.so.40`、`ucx_info` 落到系统 1.10.0）|
| HCAs | 只能选 `mlx5_0`：`UCX_NET_DEVICES=mlx5_0:1`、`UCX_IB_DEVICES=mlx5_0`。`mlx5_1/2/3` 的 `ibv_create_ah` 返回 EINVAL，多节点 ≥32 ranks 会挂 |
| MPI coll | `OMPI_MCA_coll='^hcoll,han'`（非均匀分布 7+1 会崩在 `coll/han`；模块默认只排除了 hcoll）|
| CMake | 项目要求 ≥3.30；`apprepo/tools/cmake` 已丢失（modulefile 指向的死链）。用 `~/tools/cmake-3.30.5-linux-x86_64/bin/cmake` |
| DTK cmake | 计算节点的 `/opt/rocm` 是旧 hip_hcc 树，`hip-targets.cmake` 校验 `/opt/rocm/hip/lib/libgalaxyhip.so.5.2...` 失败 → 新配置会崩。**复用已配置好的 `builds/root-instr`**，只 `make -C <build> OneFLOW/fast`（需先把 Makefile 里记录的旧 cmake 路径 sed 成 3.30.5）|
| 构建陷阱 | `git archive` 的 mtime 比对象文件旧，`make` 会跳过重编。重编前 `touch` 改过的源文件 |
| 启动方式 | **只能 `mpirun`**。`srun` 下 MPI size=1，每个 rank 解全部 zone（8×16=128 次 setup，慢 10 倍）。已加硬门禁 |

## 2. 已提交的改动（全部有 A/B 或门禁）

| commit | 内容 | 效果 |
|---|---|---|
| `47241527` | 去掉重复的 host dataBook 交换、gradient 镜像按需、interface 握手缓存；新增 per-task 计时、interface 9 段计时、200 步跨后端门禁脚本 | −27% |
| `730c52b9` | 去掉 gradient pack/unpack 的两次 pipeline drain | −0.8% |
| `b6109d07` | gradient 打包从「每 stage」改为「每轮一次」 | −4.4% |
| `89311381` | gradient receive payload 每轮上传一次 + 设备直读 unpack | −4.7% |
| `042f2c90` | 摘掉 `ilim=0` 下无意义的接口字段（limiter/dtdx.../visl/vist/dq），limiter 只填一次 | −14.5% |
| `fadb4642` | 重建的壁面状态只传 nBFaces（原来传了 13 倍数据） | recon_bcq 5.6→0.55 ms/步 |
| `fdd4ab56` | **真 Partition 任务**（METIS，1→8/16/32/64 zone 已验证）+ zone/rank 守卫 + GridConfig 可选键 | 可用 |
| `73394544` | boundary 拆成 bnd_refresh/static/kernel/ghost_copy/resize 五段 | 诊断 |
| `b4ed31d3` | boundary ghost 回拷合并成一次打包 D2H（30 次阻塞拷贝 → 2 次） | 8.7→6.6 ms/步 |

未 push：`b4ed31d3`。

## 3. 关键测量（同集群、500–1000 步、paired 或固定总工作量）

### 3.1 端到端（8 zone m6wing，8 ranks，1 DCU）

| 版本 | HIP/1000 步 | 每步 | CPU/HIP |
|---|---:|---:|---:|
| 开工 `ac594fbd` | 59.36 s | 59.4 ms | 3.89x |
| 现在 | **32.8 s** | **32.8 ms** | **6.2x**（CPU 202 ms/步）|

200 步跨后端门禁：residual `2.33e-18`、aero `0.0`。

### 3.2 单步时间 = 22 ms 固定 + 0.30 µs/cell（这是下一步的全部依据）

| cells/rank | ranks | CPU | HIP | CPU/HIP |
|---:|---:|---:|---:|---:|
| 294,912 | 1 | 63.4 s | 8.10 s | 7.82x |
| 147,456 | 2 | 35.2 s | 4.64 s | 7.58x |
| 73,728 | 4 | 20.7 s | 4.25 s | 4.87x |
| 36,864 | 8 | 13.6 s | 4.17 s | 3.26x |

固定 22 ms/步的构成（500 步、8 zone、修后）：`bnd_ghost_copy 6.6` + q halo 3 stage
+1 post 约 6 + `post_upload/download 2.7` + 三个 stage 同步点约 6 + 其它约 1。
**这些都是"按 rank × 按 stage 计数"的延迟，不随 cell 数下降**。

### 3.3 和 1D 的关系（重要，别再用错口径）

1D 端口（standalone，1 DCU vs 32-rank CPU MPI，lifecycle）：`3.85 / 5.34 / 6.80 / 9.70x`
（65k→4M），4 卡/4M `25.55x`。**1D 的 RK 同样是 host 驱动**，所以差距不在"调度在哪"，
而在每 rank 工作量：3D 在 ≥147k cells/rank 时是 7.6–7.8x，已与 1D 中段相当。
之前 "32 zone=3.0x、64 zone=1.4x 说明多卡差" 的结论**已作废**——那是把固定 294k
网格切薄导致的 per-rank 过小。

## 4. 待解决（按优先级）

### P0：RK 移植（主线，不需要新 fixture）

1. **device 权威 resident contract**：host q 镜像降为只读缓存，仅输出步刷新。
   一次消掉 `state_update` 的 5 方程 D2H 和 `bnd_ghost_copy`（−10 ms）。注意：现有
   `ResidentStateContract` 是 host 权威（跳过 D2H 会 `Invalidate()` 把陈旧 host q
   拉回设备，实测立刻报 `HIP primitive state update produced a non-physical state`）。
2. **每 stage kernel 融合**：`residual → lhs → update → boundary` 合成 1 次 launch（−6 ms）。
3. **每步一次 halo 交换**（现在 4 次：3 stage + 1 post，−6 ms）。
   目标 12–14 ms/步。

门禁顺序固定为：3 步多区 trace（`MULTIZONE_TRACE_PASS` + resident `24/0/0`）→
200 步跨后端（`residual ≤1e-12`、aero）→ CPU 五算例 normal/strict → 50 步 stability。

### P1：fixture（多卡结论的前提）

现状：**m6wing（294,912 cell，带 `m6wing.skl`）是唯一合法、可分区、可门禁的 fixture**。

- `simutask=Partition` 已可用，但只有 `partition_type=1`（ReconstructLink）能建出文件，
  且**必须有 `.skl`**；`partition_type=0`（GenerateLink）在
  `FaceTopo::SetNewFace2Node`（`codes/geometry/src/FaceTopo.cpp:335`，
  `faces.size() != GetFaceCount()`）失败。
- box 路由不可用：既有 65³ 网格 solve 时报 `62816 cells have negative vols` 并最终
  `non-physical state`（是初始/边界状态问题，不是几何），只能做 solver 趋势。
- 要拿到 ≥1M 合法分区网格，二选一：
  a) 由 fixture owner 提供 ≥1M 的 m6wing 级网格（带 `.skl`）——最省事；
  b) 修 `minimal_cube` 布局映射 + `GenerateLink` 建表。

### P2：其它已知但不紧急

- 多卡 8:1 / 跨节点：有 64-zone 能力后重测；单节点 4 卡（2 ranks/卡）曾得 11.9x，
  但那是 per-rank 变小的口径，需按"per-rank 恒定"重做。
- `bnd_resize` 报的 10 ms/步是**首调用 `hipMalloc` 被平均进短基准**，不是每步开销。
  任何 ≤100 步的基准必须先 warmup 一步，否则读数失真 10–15 ms/步。

## 5. 直接可用的脚本/开关

| 用途 | 位置 |
|---|---|
| MPI/UCX 运行环境 | `scripts/mpienv.sh`（集群 workspace）|
| 200 步跨后端门禁 | `ci/kunshan/f2-main-solver-cross-backend-residual.sh`（`ONEFLOW_XB_*`）|
| 3 步多区 trace | `ci/kunshan/f2-main-solver-multizone-trace.sh` |
| 分区 | `simutask=Partition` + `gridObj=3` + `sourceGridFileName` / `npartproc` / `partition_type` / `part_uns_file`（**这些键要写在 `control.txt` 列出的文件里**）|
| 性能开关（默认全开）| `ONEFLOW_HIP_SKIP_HOST_INTERFACE`、`ONEFLOW_HIP_GRADIENT_MIRROR_ON_DEMAND`、`ONEFLOW_HIP_CACHE_INTERFACE_HANDSHAKE`、`ONEFLOW_HIP_SKIP_HOST_FIELD_PACK`、`ONEFLOW_HIP_SKIP_UNUSED_INTERFACE`、`ONEFLOW_HIP_DEVICE_GRADIENT_UNPACK`、`ONEFLOW_HIP_BCQ_CACHE`(未用) |
| 已知不可用开关 | `ONEFLOW_SKIP_POST_INTERFACE=1`（200 步漂 5.5e-7）；`ONEFLOW_HIP_SKIP_STATE_HOST_SYNC=1`（已被移除/不可用）|

## 6. 不要做的事（都试过，有数据）

- 不要用 100 步或 10 步做端到端结论（固定开销占 20–50%）。
- 不要用 100-step 的 `2.71x`；生产口径是 1000 步、8C vs 8C+1DCU、CPU 单样本需补测。
- 不要去掉 stage kernel 后的 `hipDeviceSynchronize`：实测慢 1%，且 host 读依赖它。
- 不要删 post-process 那轮交换：它是设备打包梯度的传输通道（删了 200 步漂 5.5e-7）。
- 不要用 `srun` 启动；不要把 `partition_type=0` 当可用路径。
- 不要用 8 核 vs 64 核来说明"MPI 有问题"——求解器是纯 MPI、无 OpenMP 语义，
  rank 数=zone 数，多核闲置是正常的；要更多并行度必须有更多 zone。
