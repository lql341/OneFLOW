# OneFLOW 开发待办与衔接（living document）

## 交接摘要（先读这一段，2026-10-23）

> 本文其余按日期分节的内容是**历史证据**，不代表当前状态；当前状态、能力边界与下一步以本节为准。
> 分支模型/门禁规则见 `AGENTS.md`，集群流程见 `ci/kunshan/README.md`。

### 0. 仓库与分支（本轮已同步）

| 项 | revision | 说明 |
|---|---|---|
| `master` = `origin/master` = `upstream/master` | `b83d78c7` | 本轮 fast-forward 三端同步；含 137 个上游提交（`tools/oneflow_gui`、`grid/*`、`tools/mesh_layout`、`tests/tools`、`tests/geometry`、`vcpkg.json`、docs），**未触及 `codes/` 与 `ci/`** |
| `dev` = `origin/dev` | `762a847e`（merge）+ 本 handoff 文档提交 | `Merge branch 'master' into dev`（无冲突）；领先 `upstream/master` 264 个提交；本轮已授权并已 push `origin/dev` |
| HIP 代码已验证 revision | `95830d9b`（性能/门禁基线） | 下文数字来自该 revision；合并只新增上游 tools/grid/docs 与 3 个 tests 文件，`codes/`、`ci/` 未变 |
| 合并后 revision | `52fe404d`（Kunshan 已复验） | root HIP build + DCU preflight PASS、硬件 CTest `16/16`、8-zone 3-step trace PASS（contract `24/0/0`）、50-step stability 三路 PASS |

### 1. 目标与能力边界

**目标**：root 3D 主求解器通过 `ONEFLOW_ACCEL_BACKEND=HIP` 切换 GPU 后段，走真实主流程
（gradient→reconstruction→fused flux/residual→state update→boundary→timestep→halo），
不是 standalone port。CPU oracle 与 host 回退始终保留。

**自动启用条件**（不满足→该阶段整段 host 回退；显式请求不满足→fail-fast，不静默）：
`NS_SOLVER`、5 方程、Lax–Friedrichs（`ischeme=5`）、`ilim=0`、`vismodel=0`、
`gridLevel=0 && nGrids=1`、显式多阶段 RK（`time_integral=MULTI_STAGE`）、`nTModel=1`、
无化学、1 rank 1 zone、`nGhostCells == nBFaces`。

**已设备化（全部默认，`ONEFLOW_ACCEL_BACKEND=HIP` 即可）**：fused flux+residual、
interface/halo（传输仍 HostStaged，pack/unpack 写入 state-owned device q ghost）、
Green–Gauss 梯度、面重构（保留 legacy 的一轮交换滞后）、primitive state update +
resident residual 链、物理边界模型、resident local time-step。

**未迁移**：viscous/turbulence/SA、limiter≠0、多重网格、LUSGS、双时间、CUDA/Kokkos。

**生命周期**：q 的权威在 device `cellState`，host 镜像每 stage 由 state update 回写；
residual 只在 trace/输出/restart dump 下传，NoTrace 的残差与正性检查由设备归约标量提供；
gradient 只在 boundary-adjacent owned + ghost 上镜像（host 交换需要，也是 aero 力需要的单元）。

### 2. 关键文件地图（改这里，不要新建平行体系）

| 关注点 | 文件 |
|---|---|
| 默认策略 / 阶段开关 | `codes/accel/include/HipProductionPolicy.h` |
| HIP 生产链与 kernel | `codes/accel/src/HipFluxBackend.hip`、`codes/accel/include/HipFluxBackend.h` |
| resident state 契约 | `codes/accel/include/ResidentStateContract.h`、`codes/accel/include/AccelViews.h` |
| solver 侧调用点 | `codes/uns/src/UNsInvFlux.cpp`（gradient/recon/flux/residual）、`codes/uns/src/UNsGrad.cpp`、`codes/uns/src/UNsUpdate.cpp`、`codes/usolver/src/ULhs.cpp`、`codes/usolver/src/ULimiter.cpp` |
| 边界 | `codes/ns/src/NsSolverImp.cpp` |
| halo / ownership | `codes/task/src/InterfaceTask.cpp`、`codes/parallel/src/BoundaryExchangeContract.cpp`、`codes/global/src/InterfaceFieldProperty.cpp`、`codes/task/include/InterfaceEndpoint.h` |
| 残差/输出/气动力 | `codes/residual/src/ResidualTask.cpp`、`codes/force/src/AeroForceTask.cpp`、`codes/restart/*` |
| 计时 | `codes/accel/src/StageProfiler.cpp`（类别表 + `ScopedStageTimer`） |
| 单测 | `tests/hip_resident_state_contract_test.cpp`、`tests/euler/CMakeLists.txt` |
| 集群 runner | `ci/kunshan/f2-main-solver-{build.slurm,multizone-trace.sh,multizone-timing.sh,multizone-matrix.sh,stability.slurm,stage.slurm}` |

### 3. 开关（含兼容语义）

| 变量 | 作用 |
|---|---|
| `ONEFLOW_ACCEL_BACKEND=HIP` | 选定后端；默认 policy 打开上述全部已验证阶段 |
| `ONEFLOW_ENABLE_UNS_HIP_{BATCH,GRADIENT,RECONSTRUCTION,STATE_UPDATE,BOUNDARY,INTERFACE,TIME_STEP}` | 旧变量，现为**覆盖**：未设=auto（policy）；`0`=强制 host；`1`=请求，能力不满足则 fail-fast |
| `ONEFLOW_UNS_HIP_RESIDENT_CONTRACT=1` | 每次 gradient 前逐值比较 device(owned+ghost) vs host，打印 `RESIDENT_STATE_CONTRACT` |
| `ONEFLOW_UNS_HIP_STAGE_DIAGNOSTIC=1` | 一次性打印 gradient/recon/boundary 的决策与拒绝原因 |
| `ONEFLOW_ENABLE_BC_GRADIENT_STORE=1` | 恢复 `StoreBcGrad`（`bcdq*` 面索引梯度拷贝）；默认关，aero 力已直接读 cell gradient |
| `ONEFLOW_FAIL_ON_NONPHYSICAL=1` | 非物理 state 直接失败（device 归约标量也遵守它） |
| `ONEFLOW_RESIDUAL_TEST_OUTPUT=1` | 残差文件 `max_digits10` 精度（高精度对比用） |
| `ONEFLOW_GPU_AWARE_MPI` / `..._CONFIRMED` | DeviceDirect，**保持关闭**（见 §6） |
| `OMPI_MCA_coll='^hcoll,han'` | 昆山 MPI 必需；`han` 在不均匀分布上会崩 |
| `ONEFLOW_MZ_*` | 多 zone runner：`RANKS/STEPS/CFL/MAP_BY/HIP_DEVICES/FIXTURE_METADATA/RESIDENT_CONTRACT/...` |

### 4. 门禁证据（`95830d9b`，Kunshan DTK 26.04 / gfx906）

| 门禁 | 结果 |
|---|---|
| 8-zone 3-step trace（backend-only） | `MULTIZONE_TRACE_PASS`，每 rank 36 records，HIP vs legacy ≈2.3e-13 |
| resident contract | `records=24 failures=0 max_abs_diff=0` |
| 8-zone / 单 zone 50-step stability | 三路（legacy/cpu_batch/hip_batch）全 `exit_code=0`、`hits=0`、PASS |
| 单 zone 3-step stage trace | `STAGE_TRACE_PASS records=36` |
| **NoTrace 200-step residual vs CPU（高精度）** | `max_rel_diff = 1.094e-14`；`aero` 在输出精度内一致 |
| CPU normal / strict 五算例 | 5/5、5/5（含 viscous/turbulent） |
| 硬件 CTest / HIP gtest / resident-state 单测 | 16/16、15/15、12/12 |
| 跨节点 2×4 ranks trace | PASS（7+1 分布，见 §6） |

### 5. 性能（同 basis 规则与当前值）

**规则**：报告必须成组——同节点、同资源与绑定、同 fixture、同 `steps/warmup/repeats`；
CPU 基线只能在 `kshcnormal`（同 8 核在 DCU 节点慢 37%）；比例只与同组数据比较；
**1 DCU 槽 = 8 核**（节点 32 核/4 卡）。

| 项 | 值 |
|---|---|
| 100-step HIP（8 ranks/1 DCU，全默认阶段） | **9.08 s** 默认输出；无中间输出 **7.65 s**（cell-gather 后，2026-10-23） |
| 本轮优化前后（交臂 3 对） | 9.79 s → 9.12 s，mean 1.073x，3/3 对为正 |
| loop 求合（rank 0/4/7） | **52.4 ms/step**（优化前 57.95，−9.5%） |
| 端到端 8-zone vs CPU 8 核（`kshcnormal`） | **2.71x**（100 steps，cell-gather 后）→ **3.78x**（1000 steps：230800.4 / 61067.1 ms，同样 8C vs 8C+1DCU） |
| 同节点 8-zone"加卡"隔离（8C vs 8C+1DCU） | **3.13x**（默认输出）→ **3.24x**（无中间输出）→ **3.50x**（cell-gather 后） |
| 8-zone solver loop（profiled stage sum） | CPU `238.6 ms/step` vs HIP `45.3 ms/step` = **5.27x**（cell-gather 后）；另加约 **15.8 ms/step 未计时** host 开销 |
| 4-zone（4C vs 4C+1DCU，非 8:1） | 4.74x（41795.9 / 8822.2 ms） |
| 1-zone（1C/1 rank vs 1C+1DCU，非 8:1） | 10.01x（无中间输出：127039.8 / 12696.4 ms） |
| 1→8 rank 扩展（solver loop） | CPU 5.09x / HIP 1.22x；HIP 8-rank 绝对时间 8.14 s < 1-rank 12.70 s |
| restart `50+50` vs `100` | CPU 逐位一致；HIP 流场最大相对差约 `1.6e-14`，res/aero 输出精度内一致 |
| 固定开销 | ≈1.9 s/run（100 steps 时占 wall 约 20%；10 steps 时占 80%，故 10-step 比值无意义） |

当前 loop 分项（ms/step）：`update_residuals 27.0`（gradient 12.7 / reconstruction 9.0 /
HIP kernel 1.0 / residual_metrics 0.6）、`mpi_interface 13.1`、`update_flowfield 2.9`、
`state_update 2.5`、`calc_boundary 2.2`、`load_residuals 2.4`、`load_q 1.3`、
`calc_time_step 0.7`、`calc_lhs 0.3`。

### 6. 未完成 / 阻塞

| 项 | 状态与原因 |
|---|---|
| **DeviceDirect** | probe 通过但实测慢 1.59x（MPI 内部仍 host staging）→ 保持关闭 |
| **多卡 8:1 核卡比** | 缺 ≥64-zone fixture；`simutask=Partition` 在本 build 抛 "not implemented"，需外部 METIS 分区或 owner 提供 |
| **跨节点 8:1 比值** | 正确性 PASS，但分配固定 7+1、2 卡 4 核/卡；portal 不接受 `--ntasks-per-node` |
| viscous/RANS/limiter/multigrid/LUSGS | 未迁移，host 回退 |

### 7. 下一步（按序，含验收）

1. **接口紧凑布局**：`swap_upload+swap_download ≈ 4.3 ms/step` 是 `SwapBcGrad` 为接口 storage 做的随机 gather/scatter；让交换直接消费 endpoint 布局。验收：CPU 5/5、3-step trace、200-step residual、A/B ≥3%。
2. **gradient_mirror 3.5 / reconstruction 9.0** 的进一步去搬运（不要动数值语义与时序）。
3. **64-zone fixture**（并行请求）→ 16-zone 双卡 → 多卡 8:1。
4. 受限路径闭环后再选生产能力扩展：层流 NS → RANS/SA。

### 8. 复现（Kunshan）

```bash
# 同步源码（scnet-hpc skill）
git archive --format=tar.gz --prefix="OneFLOW-<rev>/" <rev> -o /tmp/OneFLOW-<rev>.tar.gz
python3 <skill>/scripts/scnet.py upload /tmp/OneFLOW-<rev>.tar.gz /public/home/luql/.scnet-hpc/oneflow/incoming
# 构建 + 硬件 CTest（DCU 队列）
sbatch --export=ALL,ONEFLOW_F2_BUILD_SOURCE_DIR=<src>,ONEFLOW_F2_BUILD_DIR=<build> f2-main-solver-build.slurm
# 门禁（backend-only：不要设 ONEFLOW_ENABLE_UNS_HIP_*）
ONEFLOW_MZ_RUN_ROOT=<run> bash f2-main-solver-multizone-trace.sh        # 3-step trace + contract
ONEFLOW_F2_STABILITY_RUN_DIR=<run> bash f2-main-solver-stability.slurm  # 50-step 三路
bash f2-main-solver-multizone-timing.sh                                  # 100-step 配对/交臂
```
所有原始日志、私有 fixture、ownership manifest 留在集群 artifacts，不进仓库。

## 2026-10-23 受限 inviscid HIP 收口：restart 等价性与多配置端到端

**结论**：交接文档描述的七阶段受限 inviscid 路径（fused flux/residual、interface、
gradient、reconstruction、state update、boundary、resident time-step）在 `52fe404d`
上仍是 backend-only 默认，本轮没有发现新的 HIP 化缺口，也没有修改数值代码。此前
唯一未实测的正确性项——restart 等价性——已补齐；同时给出 1/4/8-zone 三种分区
数量和 1D Euler 五个网格规模的同 basis 数据，用来避免只引用单点。

### 复核门禁（backend-only，不带 `ONEFLOW_ENABLE_UNS_HIP_*`）

| 项 | 结果 |
|---|---|
| 8-zone 3-step trace | `MULTIZONE_TRACE_PASS`，8/8 rank 有 solver work，每 rank 36 records |
| resident contract | `records=24 failures=0 max_abs_diff=0` |
| HIP vs legacy 最大绝对差 | `2.2837e-13`（face/reconstruction 量级，metadata 差为 0） |
| 100-step CPU/HIP | 全部 `exit_code=0`，`ONEFLOW_FAIL_ON_NONPHYSICAL=1`，rank utilization 全 PASS |
| 环境 | DTK 26.04 / `gfx906`；CPU 与 HIP 同一作业内交替运行，固定 CFL `0.01` |

### restart 等价性（1-zone，50 步 → 重启 → 再 50 步 vs 连续 100 步）

| backend | 最终 `flow.dat` | 最终 `res.dat` | 最终 `aero.dat` | 判定 |
|---|---|---|---|---|
| CPU | SHA-256 完全一致 | 相对差 `0`（输出精度内） | 相对差 `0`（输出精度内） | PASS |
| HIP | 非逐位一致；20 个数组中有 10 个流场数组不同 | 相对差 `0`（输出精度内） | 相对差 `0`（输出精度内） | PASS |

HIP 流场数组的最大相对差为 `1.63e-14`（同一基线的另一次独立运行得到 `1.51e-14`；绝对差同量级），残差数组逐位一致。
这满足交接文档的 restart 判据（`res` ≤1e-12、`aero` ≤1e-10），但**不能**写成
“HIP restart 逐位一致”；逐位一致目前只在 CPU 路径成立。已加入可复现 runner：
`ci/kunshan/f2-main-solver-restart-equivalence.sh`（env 驱动，不内置集群路径）。

### 端到端：分区数量扫描（m6wing，100 steps，warmup 1，repeats 2，fixed CFL 0.01）

同节点行在同一 DCU 节点、同一作业内交替运行；`zone8` 的规范行来自独立的 CPU
队列作业（`kshcnormal`，8 核），与 HIP 行不是同一节点。表中为 2 个正式 repeat
的均值（warmup 不计）。`zone8` 是规范 8:1 槽；`zone4`/`zone1` 分别是 4 核/DCU 与
1 核/DCU，只用于观察“分区数量/单点差异”，不能当作 8:1 核卡比。

| 配置 | 资源 | CPU mean | HIP mean | CPU/HIP |
|---|---|---:|---:|---:|
| 8-zone / 8 ranks | 8C（`kshcnormal`） vs 8C+1DCU | 21599.9 ms | 9079.6 ms | **2.38x** |
| 8-zone / 8 ranks | 同节点 8C vs 8C+1DCU（隔离加卡） | 28400.8 ms | 9079.6 ms | **3.13x** |
| 4-zone / 4 ranks | 4C vs 4C+1DCU（非 8:1） | 41795.9 ms | 8822.2 ms | 4.74x |
| 1-zone / 1 rank | 1C vs 1C+1DCU（非 8:1） | 129670.3 ms | 13029.2 ms | 9.95x |

样本（正式 repeat）：8-zone CPU `21686.1 / 21513.8`（CPU 队列）、`28966.9 / 27834.8`
（DCU 节点）；8-zone HIP `9303.4 / 8855.8`；4-zone CPU `41693.7 / 41898.0`、HIP
`8790.6 / 8853.8`；1-zone CPU `129967.2 / 129373.3`、HIP `13043.3 / 13015.0`（ms）。
分区数量越少、CPU 侧活跃核越少，比值越大；因此 1-zone 的 9.95x **不能**外推成
8:1 槽的加速比。规范口径应引用 8-zone 的 `2.38x`（跨队列）或 `3.13x`（同节点隔离）。

### 8-rank 损耗的实测（不是推测）

同一套 m6wing 294,912 cell、100 steps、无中间输出、各 2 个正式 repeat：

| 配置 | CPU | HIP | CPU/HIP |
|---|---:|---:|---:|
| 1-zone / 1 rank | 127039.8 ms | 12696.4 ms | **10.01x** |
| 8-zone / 8 ranks | 20714.5 ms | 8143.3 ms | **2.54x** |

solver loop（各 rank 顶层阶段求和，再对 rank 取均值）：

| 配置 | CPU ms/step | HIP ms/step | CPU/HIP |
|---|---:|---:|---:|
| 1-zone / 1 rank | 1217.88 | 59.86 | **20.35x** |
| 8-zone / 8 ranks | 239.27 | 49.21 | **4.86x** |

1→8 rank 的实测扩展：

| | CPU | HIP |
|---|---:|---:|
| wall | 127039.8 → 20714.5 = **6.13x** | 12696.4 → 8143.3 = **1.56x** |
| loop | 1217.88 → 239.27 = **5.09x** | 59.86 → 49.21 = **1.22x** |
| 8 核/rank 效率 | 64% | 15% |

**结论：8-rank HIP 的绝对时间更快（8.14 s < 12.70 s），不是“8 rank 让 HIP 变慢”。**
比值从 10.0x 掉到 2.54x，是因为 CPU 用 8 核拿到 5.09x loop / 6.13x wall 的扩展，
而 1 张 DCU 被 8 个 rank 共享只拿到 1.22x loop / 1.56x wall。8 个 rank 把
36,864 cell/rank 的 host 工作并行化，但 device 的总工作量没变，8-rank 的
`mpi_interface` 还占 10.1 ms/step；所以这是**相对扩展效率损失**，不是绝对损耗。
1-zone 的 10.0x 端到端也含约 6.7 s 固定开销（12.70 s wall - 5.99 s loop），
其 loop 比值 20.35x 才是纯 solver 比值。

### 输出/固定开销与 solver loop 的分解

上面的 `2.38x` 是包含 `nFieldSave=50`、`nResSave=1`、`nForceSave=1` 的端到端值。
把三个保存间隔都设为 1000（100 步内不写中间 restart；最终 postprocess 仍写 aero/res，restart 由 `nFieldSave` 门控，本次不再写）
后，同一 8-zone、8-rank、100-step、2-repeat basis：

| 项 | CPU | HIP | CPU/HIP |
|---|---:|---:|---:|
| 默认输出（中间 restart/aero） | 21599.9 ms（CPU 队列）/ 28400.8 ms（同节点） | 9079.6 ms | 2.38x / 3.13x |
| 无中间输出 | 20714.5 ms（CPU 队列）/ 26414.2 ms（同节点） | 8143.3 ms | **2.54x / 3.24x** |
| solver loop（各 rank 的顶层阶段求和） | 239.3 ms/step | 49.2 ms/step | **4.86x** |

结论：端到端与 loop 的差距主要是固定开销（HIP wall `8143 ms` vs loop `4921 ms`，
约 `3.2 s`，占 39%）；去掉 restart 写出只把 `2.38x` 提到 `2.54x`，所以它不是
主因，但说明写出成本约 `0.9 s`（HIP）/`2.0 s`（CPU）。
固定开销来自 MPI/设备初始化、31 MB 网格读入、最终 restart/aero 写出和进程启动。
1D Euler 4M 的整个 lifecycle 只有约 `0.73 s`，没有逐 rank 网格读入/MPI/输出，因此
它的比值可以很高。

### 16-zone 分区尝试：能划分，但 ad-hoc 分区不是合法 HIP fixture

用 `simutask=Grid`、`gridObj=3`、`npartproc=16` 在 CPU 节点上约 25 s 生成了
`m6wing-16.ofl`（16 zone，每 zone 约 1.84 万 cell，总 cell 数仍是 294,912）。
这证明**划分本身不是阻塞**，4/8-zone fixture 也是同一流程生成的。但把该 ad-hoc
16-zone 直接交给 HIP 路径（16 ranks/1 DCU）会退化：

| 16-zone ad-hoc | 100 steps | `mpi_interface` | `calc_time_step` |
|---|---:|---:|---:|
| CPU 16 ranks | 13047.7 ms | 5.52 ms/step | 2.47 ms/step |
| HIP 16 ranks/1 DCU | 131604.0 ms | 203.7 ms/step | 32.3 ms/step |

HIP 的 `mpi_interface` 和 `calc_time_step` 没有进入 8-zone 已验证的 resident/契约
路径；最可能的原因是该 ad-hoc 16-zone 网格没有经过 8-zone fixture 的
ownership/interface manifest 导出与验证。在取得合法 manifest 前，这条数据
不能用于性能结论，已从数据表排除。合法扩展必须先用
`ONEFLOW_MZ_FIXTURE_EXPORT_DIR` 导出 `rank-<rank>.json`、合并、用
`verify-multizone-fixture.py` 验证，再跑 trace/stability；这也解释了为什么交接
文档要求“≥64-zone fixture”而不是“随手 partition 一下”。
另外，增加 zone 数只会把同样的 294,912 cell 切得更碎，**不会增加每个 DCU 的
计算量**；要提高 3D 比值，需要更大的网格（更多 cell/DCU），而不是更多 zone。

### 3D 与 1D 的可比性：目标不是 100–300x

1D 报告里的 `157–293x`（stateful）是**单线程 CPU vs 1 DCU**的 compute-path
微基准；同一份 1D 报告里，1 DCU vs **32-rank CPU MPI** 的应用级 lifecycle 是
`3.85x / 5.34x / 6.80x / 9.70x`（65k → 4M）。3D 的 `2.38–2.54x` 应该和后者比，
而不是和 `100–300x` 比。

当前 3D 的位置：

- **solver loop `4.86x`**：已经落在 1D 应用级 `3.85–9.70x` 的下半段；
- **端到端 `2.54x`**：被约 `3.2 s` 固定开销和 host-staged 阶段拉低；
- 每 rank 只有 36,864 cell（8-zone/8-rank），而 1D 4M 是 4M cell/DCU；1D 的
  `65k → 4M` 比值从 3.85x 升到 9.70x，说明规模是最强的单一杠杆；
- loop 内 HIP 融合 kernel 只有约 `0.9 ms/step`，`gradient 12.8`、`reconstruction 10.4`、
  `mpi_interface 10.1`、`load_residuals 2.1`、`load_q 1.0` 仍是 host/契约编排；
  `gradient_mirror 3.3`、`halo_ghost_sync 3.1`、`swap_upload 3.0`、`swap_download 1.7`
  是接口紧凑布局的直接目标。

因此：**3D 有希望接近 1D 的应用级 4–10x，但前提是更大网格 + 去掉 host staging；
不可能也不应该以 1D 的 100–300x（单线程分母）为目标。** 当前 2.54x 到 4–6x 的
主要缺口是固定开销（~3.2 s）、`mpi_interface`/`SwapBcGrad` 和 gradient/recon
的 host 编排，而不是 HIP kernel 算力。

### 补充：1D Euler 网格规模扫描（serial CPU vs 1 DCU，非 3D 主求解器）

为回答“单点差异”，在同一 DCU 节点上用 standalone 1D Euler stateful backend 跑了
五个网格规模（100 steps、warmup 1、repeats 2，`final_max_abs_error` 全为 0）。
该二进制由当前 `52fe404d` 源码在本轮重新构建；下表是 **compute-path 微基准**：
CPU 是单线程，HIP 是 1 张 DCU，不是 32-rank CPU 的应用级对比，也不能替换上面的
3D 主求解器数据。它与 2026-09-13 报告中的历史 CPU 基线不可混用（当前源码的单线程
CPU 路径更慢，HIP advance 基本不变）。

| nx | CPU advance (ms) | HIP advance (ms) | HIP lifecycle (ms) | advance 比值 | CPU advance / HIP lifecycle |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 1063.4 | 9.44 | 10.50 | 112.7x | 101.3x |
| 65,536 | 3836.6 | 15.04 | 18.57 | 255.2x | 206.6x |
| 262,144 | 13171.8 | 46.36 | 52.80 | 284.1x | 249.5x |
| 1,048,576 | 50192.5 | 168.48 | 188.48 | 297.9x | 266.3x |
| 4,194,304 | 216107.9 | 656.53 | 729.57 | 329.2x | 296.2x |

`lifecycle` = `create + upload + advance + download`，与 advance 同为 repeats 总和。
随 `nx` 增大，固定开销被摊薄，比值从 101.3x 单调升到 296.2x；小网格的 101.3x
不能代表大网格。

### 加密 3D 网格的尝试（2026-10-23，未形成可用数据点）

为了直接回答“是不是网格太小”，本轮尝试了三条路，均未得到可用的 1M+ cell 3D
性能点：

- **布局生成器**：把 `minimal_cube` 的 `Dim` 从 3 改到 65，生成 65³（262,144 cell）
  网格时出现 `62,816 cells have negative vols`；该布局的 face/line 定向在细网格上
  不自洽，3³ 能过只是规模小。该网格不能用于求解。
- **Plot3D 规则 box**：用规则 [0,1]³ 坐标写 `.grd`/`.inp`，再转成 `.ofl`；
  **网格生成成功**：65³ = 262,144 cell、97³ = 912,673 cell、129³ = 2,097,152 cell
  （边界面 24,576 / 55,296 / 98,304），`.ofl` 约 28 MB / 98 MB / 220 MB。
  之前用 `nFieldSave/nResSave/nForceSave=1000` 时 flow case 在 setup 后不进入
  迭代（Release exit 1，RelWithDebInfo exit 0 且无 `res.dat`）。**根因已定位为
  该 box case 的 save cadence 配置**：把 `nResSave`/`nForceSave` 设为 1 后，
  三套网格都正常迭代 50 步，残差约 `1e-19`（均匀流），65³/97³/129³ 的 CPU/HIP
  曲线见下一节。调试构建过程中还发现并修掉了当前 CPU 构建的真实回归：
  `NsSolverImp.cpp` 在 `ONEFLOW_ENABLE_HIP` 外调用了
  `HipProductionFeatureEnabled`，导致 CPU 构建在 `NsSolverImp.cpp:270` 直接
  编译失败（已把 `HipProductionPolicy.h` 改为无条件包含）。
- **8-zone 单 rank**：想用 8 个 local zone 把每 DCU 的 cell 数提高到 294,912，
  CPU 路径直接 exit 1；HIP 10 steps 为 15.9 s（1.59 s/step），也不是可用的
  8:1 对照。

因此，**目前没有一个经过 trace/positivity 验证的 1M+ cell 3D 网格**。m6wing 仓库
里只有一套 294,912-cell 网格，没有 refinement 工具或原始 Plot3D 源；要严格证明
规模效应，需要外部提供更大的 m6wing/其他 3D 网格，或把 box 的 BC/初始化链修通。

但现有数据已经说明“网格太小”不是全部原因：同一套 294,912 cell，1-zone/1-rank
（1C vs 1C+1DCU）是 **9.95x**，8-zone/8-rank（8C vs 8C+1DCU）是 **2.54x**。
差别来自 8 个 rank 共享 1 DCU、MPI/halo/host staging 和约 3.2 s 固定开销，而不是
cell 数从 295k 变成别的值。1D 的 65k→4M 比值从 3.85x 升到 9.70x 说明规模确实
是杠杆，但 3D 当前更大的杠杆仍是接口紧凑布局与 gradient/recon 去搬运。

### 加密网格实测：65³ / 97³ / 129³（1-zone，1C vs 1C+1DCU，50 steps）

同一套 box 几何、1 个 zone、1 个 rank，1 核 CPU vs 1 核 + 1 DCU；`nResSave=1`、
`nForceSave=1`、`nFieldSave` 大于步数；3 个正式 repeat 均值，50 steps：

| N | cells | CPU mean | HIP mean | 端到端 CPU/HIP | CPU loop ms/step | HIP loop ms/step | loop CPU/HIP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 65 | 262,144 | 53,922.1 ms | 5,661.1 ms | **9.53x** | 985.25 | 43.46 | **22.67x** |
| 97 | 912,673 | 182,400.1 ms | 16,725.4 ms | **10.91x** | 3432.23 | 136.05 | **25.23x** |
| 129 | 2,097,152 | 503,404.2 ms | 34,607.9 ms | **14.55x** | 9631.64 | 308.17 | **31.25x** |

观察：

- 网格从 262k 增到 2.1M（8x），**loop 的 CPU/HIP 比值从 22.67x 升到 31.25x**，
  端到端从 9.53x 升到 14.55x；规模效应确实存在，但远没有 1D 的 100–300x 那么夸张。
- 端到端低于 loop，是因为 fixed overhead 随网格增大：65/97/129 的 HIP wall − loop
  分别为约 3.5 / 9.9 / 19.2 s，CPU 为约 4.7 / 10.8 / 21.8 s；主要是网格读入、
  初始化和输出，不是 device kernel。
- 这是 **1 core vs 1 DCU** 的口径，不是 8:1 槽。8:1 槽仍是 8 ranks 共享 1 DCU 的
  `2.54x`；把网格放大到 2.1M 且仍用 1 zone/1 rank 只能证明“单卡单 rank 下规模
  越大比值越高”，不能直接外推成 8 卡/8:1 的线性加速。
- 3D 2.1M 的 loop 31.25x 仍低于 1D 4M 的 296x，原因是 3D 每 cell 有
  gradient/reconstruction/boundary/MPI/host 编排，且 1D 路径没有这些；这不是
  同一算法。

### 全链路审计：哪些是 device，哪些还是 host

“受限 inviscid HIP”不等于“3D solver 全链路 GPU 化”。当前实际分层：

| 链路段 | 状态 | 证据/位置 |
|---|---|---|
| 网格读入、分区、wall distance | host | `FieldSimuLoadGrid`/`PrepareWallDist`；`vismodel<=1` 跳过 wall distance |
| solver/state 创建与初始 q | host + H2D | `FieldSimuCreateSolvers`/`InitFlowField` |
| **时间推进调度（RK stage 循环）** | **host（CPU）** | `TimeIntegral::RungeKutta` 使用 `CpuEulerDomainBackend` + `AccelBackendKind::CPU`，每个 stage 回调 host task |
| gradient | device | `UNsGrad`/`CalcHipGradient`，但 ghost 镜像/`SwapBcGrad` 仍 host |
| reconstruction | device | `ReconstructFaceValues`，跨 zone ghost gradient 仍 host mirror |
| inviscid flux + residual | device | fused `CalcAndAddReconstructedFaceFlux` |
| state update / residual chain | device | `UpdatePrimitiveState` + resident residual |
| physical boundary | device（受限类型） | `UpdateBoundaryState`；host `bcQ`/ghost 镜像 |
| interface/halo | **host-staged** | `InterfaceTask`/`BoundaryExchangeContract`；MPI 仍是 host buffer |
| limiter | host / 未迁移 | `ilim!=0` 直接回退 |
| LUSGS / multigrid / dual-time | host / 未迁移 | `time_integral!=MULTI_STAGE`、`nGrids>1` 回退 |
| viscous/turbulence/SA | host / 未迁移 | `vismodel!=0` 回退 |
| restart/输出/气动力 | host | `RestartTask`、`ResidualTask` 输出、`AeroForceTask`（D2H 后） |

所以准确说法是：**受限 inviscid 的“空间离散 + 面通量 + 残差 + 状态更新 + 边界 + local time-step”已 device 化；时间推进调度、接口传输、IO/restart、气动力、limiter/LUSGS/multigrid/viscous 仍是 host 或未迁移。** 这不是完整 GPU-resident 的全链路。

**移植方式是否最优**：当前是“正确性优先、逐 stage host 编排 + resident device state”的混合方案，不是性能最优：

- 每个 stage 由 host 调度并频繁 `hipDeviceSynchronize`，没有 HIP graph/多 stream 重叠；
- interface 仍是 host pack/MPI/host unpack，`swap_upload+swap_download` ≈ 4.7 ms/step，`gradient_mirror` ≈ 3.3，`halo_ghost_sync` ≈ 3.1；
- 融合 kernel 只有约 0.9 ms/step，说明 device 算力不是瓶颈，瓶颈是 host 编排/传输/fixed overhead；
- 没有针对 DTK/gfx906 做 occupancy/带宽/`__launch_bounds__`/LDS/vector load 调优；但即使 kernel 再快，也压不过 host staging。

下一步优先级应是：1) 把 RK stage 调度搬到 device 或至少用 HIP graph 串起来；2) 接口紧凑布局 + 避免 `SwapBcGrad` 的 host gather/scatter；3) gradient mirror/ghost sync 去搬运；4) 在以上之后再用 DTK 文档/rocprof 调 kernel（occupancy、HBM 带宽、wavefront 64、LDS、`hipMemcpyAsync`/pinned）。直接上 DTK 微优化不会把 2.54x 变成 4–6x。

### 1D vs 3D 的实现效率差异（实测 + 代码级原因）

先把“算法工作量”和“实现效率”分开。1D 是一阶 Rusanov + SSP-RK3、3 方程、
无 gradient/reconstruction；3D 是 5 方程、Green–Gauss gradient、二阶
reconstruction、boundary、interface。3D 每 cell 更贵是应该的；但实测贵得更多：

| 指标 | 1D persistent（4M cell） | 3D box（2.1M cell，1 rank） | 3D m6wing（294k，8-zone loop） |
|---|---:|---:|---:|
| HIP 吞吐 | 3.28 ms/step → 1.28 G cell/s（0.78 ns/cell） | 308.17 ms/step → 6.8 M cell/s（147 ns/cell） | 49.21 ms/step（总 294k）→ 5.98 M cell/s（167 ns/cell） |
| CPU 吞吐 | 1080.5 ms/step → 3.88 M cell/s（258 ns/cell） | 9.63 s/step → 0.218 M cell/s（4.59 µs/cell） | 239.27 ms/step → 1.23 M cell/s（812 ns/cell） |
| CPU/HIP | 329x（advance）/296x（lifecycle） | 31.25x | 4.86x |

3D HIP 每 cell 比 1D HIP 慢约 **188x**，但 3D CPU 每 cell 只比 1D CPU 慢约
**17.8x**；说明 3D 的 HIP 实现相对 CPU 明显更差，不只是算法更复杂。代码级原因：

1. **gradient 用 face-based `atomicAdd`**：`GradientAccumulateKernel` 对每个面、
   每个方程、每个分量做 6 次全局 atomicAdd（5 方程 ≈ 30 次/内面），再加
   `GradientZero/Normalize/CopyBoundary` 三个 kernel。1D 没有 gradient，也没有
   atomic；这是 3D 独有的串行化/带宽开销。
2. **residual 也用 face-based scatter + atomicAdd**：`ScatterFaceFluxEulerKernel`
   每个内面对 5 方程各 2 次 atomicAdd（≈10 次/面）；1D 的 `ResidualUpdate`
   是规则的 cell stencil，完全无 atomic。
3. **face-based gather 不合并**：reconstruction/flux 每个线程处理一个 face，
   读 `leftCell[face]`/`rightCell[face]` 指向的 cell（间接、随机），而 1D 的
   `il=face-1, ir=face` 是连续访问。
4. **寄存器压力**：`ReconstructFaceEulerKernel` 用 `leftValue[5]`、
   `rightValue[5]`、`leftCandidate[5]`、`rightCandidate[5]`（20 个 double）+
   多组 gradient 读取；1D 的 `Face` 只用 3 分量和少量标量，occupancy 更高。
5. **kernel 粒度/同步**：3D 一个 stage 要 zero/accumulate/normalize/copy gradient、
   reconstruct、flux、scatter、boundary、state 等多个 launch，并在 host 侧
   `hipDeviceSynchronize`；1D NoTrace 每步只有 `Face` + `ResidualUpdate` 两个
   kernel，且整个 100 步在 device 上连续跑。
6. **host 编排**：3D 每个 RK stage 由 host `SingleSolverSingleGridTask` 调度，
   经过 `MRField`/virtual/task registry；1D 是 persistent device state，没有
   host stage loop。
7. **MPI/interface**：3D 8-zone 有 35,624 个 interface face，host pack +
   `SwapBcGrad` + MPI；1D halo 只是一个 rank ring。
8. **每 DCU 工作量**：3D 8:1 每 rank 36,864 cell；1D 4M 是 4M cell/DCU；
   launch/sync/固定开销摊不薄。

所以“3D 算法并行度/向量化不如 1D”这个怀疑是有代码和实测依据的，尤其是
**face-based atomic scatter** 和 **host per-stage 编排**。但也不能简单说
“3D 算法本身差”：3D 的 gradient/reconstruction/boundary 是 1D 没有的算法步骤，
其中一部分成本是合理的；需要用“同算法、不同实现”的 A/B 才能把两部分完全分开。

### 主杠杆落地：cell-based gather 替代 face-based atomic scatter（`9b5934bd`）

按 1D 的 `Face → ResidualUpdate` 模式改 3D：

- 新增 cell-to-face CSR（`cellFaceOffsets/cellFaceIds`），每个 topology 建一次；
- gradient：face 并行 `atomicAdd`（5 方程 × 6 分量 ≈ 30 次/内面）+ zero/normalize
  三个 kernel → **cell 并行 gather**，每个 cell 只写一次，无 atomic；
- residual：fused flux 只写面通量，新增 cell 并行 gather（带 explicit boundary
  mask），去掉 `ScatterFaceFluxEulerKernel` 的 ≈10 次 atomicAdd/内面；
- primitive 与 reconstructed 两条 fused 路径都改为“face flux + cell gather”。

正确性：HIP smoke 全 PASS、HIP CTest `16/16`、8-zone 3-step trace
`MULTIZONE_TRACE_PASS`、resident contract `24/0/0`、HIP vs legacy 最大绝对差
`1.806e-13`。

A/B（同节点、100 steps、无中间输出；先 warm 两个二进制，再 3 对同节点交错）：

| 项 | 旧 face-atomic | 新 cell-gather | 变化 |
|---|---:|---:|---:|
| **paired wall（warm）** | 7843.3 ms | **7716.5 ms** | **−1.6%（3/3 对为正）** |
| 8-zone HIP solver loop | 49.21 ms/step | **45.27 ms/step** | −8.0% |
| `gradient_kernel` | 4.28 ms/step | **0.86 ms/step** | −80% |
| 65³ box loop 比值 | 22.67x | **24.34x** | +7.4% |
| 129³ box loop 比值 | 31.25x | **33.15x** | +6.1% |

> 之前一版跨作业的 `HIP wall 8143.3 → 7649.6 ms（−6.1%）` 混入了节点/冷启动
> 差异；同节点 warm 后的配对结果是 **−1.6%**，更可信。loop 的 −8% 是真实算法
> 收益，但端到端被约 3.5 s fixed overhead 稀释。

早期跨作业 A/B（仅作趋势，不作正式结论）：

| 项 | 旧 face-atomic | 新 cell-gather | 变化 |
|---|---:|---:|---:|
| 8-zone HIP wall | 8143.3 ms | **7649.6 ms** | −6.1% |
| 8-zone HIP solver loop | 49.21 ms/step | **45.27 ms/step** | −8.0% |
| `gradient_kernel` | 4.28 ms/step | **0.86 ms/step** | −80% |
| 8-zone 同节点 CPU/HIP | 3.24x | **3.50x** | +8% |
| 8-zone CPU 队列 CPU/HIP | 2.54x | **2.71x** | +6.6% |
| 65³ box HIP wall / loop 比值 | 5661.1 ms / 22.67x | **5497.4 ms / 24.34x** | −2.9% / +7.4% |
| 129³ box HIP wall / loop 比值 | 34607.9 ms / 31.25x | **33436.8 ms / 33.15x** | −3.4% / +6.1% |

`gradient` 父项从 14.92 → 11.56 ms/step，`gradient_kernel` 4.28 → 0.86；
`reconstruction` 7.15、`mpi_interface` 12.74、`gradient_mirror` 3.85、
`halo_ghost_sync` 3.41、`swap_upload+swap_download` 5.90 成为新的前几项。
**atomic 是真实瓶颈之一，但去掉后 loop 只降 8%；端到端只降 3–6%，因为 fixed
overhead 和 interface host staging 还在。** 下一步主杠杆是接口紧凑布局 /
`SwapBcGrad` 去 host，以及 reconstruction 的寄存器/face-based 访问；RK 仍按
你的要求先不做。

### RK 移植的预期收益

当前 `TimeIntegral::RungeKutta(context)` 仍是 host：它用 `CpuEulerDomainBackend`
+ `AccelBackendKind::CPU` 驱动 stage 回调，每个 stage 调
`LOAD_RESIDUALS → UPDATE_RESIDUALS → CALC_LHS → UPDATE_FLOWFIELD → CALC_BOUNDARY`
（多 zone 还有 `UPDATE_INTERFACE_DATA`）。8-zone HIP 的每步顶层耗时里：

- `load_residuals` 2.1 ms、`calc_lhs` 0.3、`update_flowfield` 2.8、
  `calc_boundary` 2.8，合计约 **8.0 ms/step（16%）**；
- `update_residuals` 27.8（gradient/recon/flux 的 device 工作）和
  `mpi_interface` 10.1 不是 RK 调度本身。

因此**只把 RK stage 调度搬到 device**，乐观估计省掉一半的 host 调度/同步，
大约 **4–8 ms/step**，loop 49.2 → 41–45 ms（8–17%），端到端 2.54x 大约
2.7–2.8x；若同时把 3 个 stage 的 gradient/recon 融合、避免重复 zero/mirror，
收益才会更大。它值得做一个原型，但不是把 2.54x 推到 4–6x 的主杠杆；主杠杆仍是
gradient/recon 的 atomic/gather 和 interface host staging。

### 其余瓶颈的实验状态与预期

| 瓶颈 | 状态 | 实测/预期 |
|---|---|---|
| gradient/residual atomic | **已做**（cell-gather） | loop −8.0%，paired wall **−1.6%**；gradient kernel −80% |
| reconstruction 寄存器 | 已做（去掉 2×5 冗余数组） | opt2 中 reconstruction 7.15→6.86 ms/step；端到端在噪声内，未宣称 |
| interface batch（x/y/z 共用 endpoint id） | **试过，未保留** | 无收益（`swap_upload 3.88→4.69`，`swap_download 2.02→1.72`），已回退 |
| interface 紧凑布局 / `SwapBcGrad` 去 host | 未做（结构性改动） | 预期 loop −4–8 ms/step（9–18%），端到端因 fixed overhead 约 −2–5% |
| reconstruction+flux fusion | 未做 | 预期 loop −1–3 ms/step（2–6%） |
| gradient mirror / halo ghost sync | 未做 | 预期 loop −1–2 ms/step |

**全部做完的修订预期**（不要把 2.54x 直接推成 4x）：8-zone loop
45.3 → 约 **37–40 ms/step**（−12–18%）；paired wall 7.72 s → 约 **7.3–7.5 s**
（−3–5%）；CPU 队列比值 2.71x → 约 **2.8–2.9x**。原因是 HIP wall 里约 **3.5 s
（46%）是 fixed overhead**，interface 还有约 18.6 ms/step；只改 loop 不改
fixed overhead/interface，端到端提升会被按比例稀释。129³ 1-zone 的 fixed overhead
约 19 s，情况更严重。

### 第二轮：interface 微优化实测（收益递减）

- `HXWriteSubData/HXReadSubData` 的 per-equation staging 改为 thread-local
  复用：paired warm A/B `7883.2 → 7851.2 ms`（**1.0041x**，在噪声内）。
- x/y/z 批量 upload/download：`swap_upload 3.88→4.69`、`swap_download
  2.02→1.72`，总 loop 无收益，已回退。
- reconstruction 去冗余数组：reconstruction `7.15→6.86 ms/step`，端到端在
  噪声内。
- 8-zone cell-gather 的 paired warm 结果是 **1.0164x**（3/3 对为正），
  不是跨作业的 6.1%。

结论：**继续做 loop 微优化的边际收益已经很小**。剩下的 interface 紧凑布局
是结构性改动（storage/device pack 顺序），预期 loop −4–6 ms/step，但按当前
fixed overhead 只会带来约 **1–2% wall**；更大的 wall 杠杆是约 **3.5 s（46%）**
的 fixed overhead（129³ 1-zone 约 20 s）和 interface 的 host 交换本身。若目标是
4x，需要先做 fixed-overhead profiling/削减，而不是再调一个 kernel。

### 1000-step 基准：100-step 的 2.71x 是基准问题

100-step 的 `wall−loop` 差值约 3.2 s 曾被当成“固定开销”，但 1-step 实测只有
**1.83 s（old）/1.95 s（new）**；3.2 s 里还包含约 1.4 s 的**未计时 per-step**
开销。1000-step 同节点 paired warm A/B（8 ranks，1 DCU，无输出）：

| 项 | old face-atomic | new cell-gather |
|---|---:|---:|
| 1-step wall（含启动/收尾） | 1827.2 ms | 1947.0 ms |
| 1000-step wall（3 对均值） | 62589.6 ms | **61067.1 ms（−2.4%）** |
| per-step wall | 62.59 ms | **61.07 ms** |
| profiled stage sum | 49.21 ms | **45.27 ms** |
| **未计时 per-step** | **13.38 ms（21%）** | **15.80 ms（26%）** |

CPU 8 核 1000-step：**230800.4 ms**。

**host 链第一刀（`afb7e8d6`）**：把 HIP gradient 的 interface 交换改成
device pack/unpack——send 端在设备上按 `[component][equation][face]` 打包，
host 直接拷入 interface storage；recv 端 host 复制后直接 scattered 到设备
ghost gradient，跳过 host cell gather/scatter 和单独的 ghost H2D sync。
host mirror 仍保留给 aero 力。

- 1000-step paired warm（vs cell-gather）：`61491.9 → 60186.3 ms`（**1.0217x**）；
- vs 原始 face-atomic：`62589.6 → 60186.3 ms`（**−3.8%**）；
- CPU 230800.4 / 60186.3 = **3.835x**（原始 3.69x）；
- 3-step trace PASS、contract `24/0/0`、HIP CTest 16/16。
- `ONEFLOW_HIP_GRADIENT_PACKED=0` 可回退 host swap。

这说明 host 链可以一刀一刀切，但每刀约 1–2 ms/step（1–2% wall）；剩下的
`mpi_interface 12.74`（q 交换/MPI）和约 15.8 ms/step 未计时 host 块才是
下一刀的主目标。因此：

- 100-step 的 CPU/HIP `2.71x` 被 1.8 s 启动/收尾和 CPU 节点差异放大了；
- **1000-step 的实际 8:1 wall 比值是 230800.4 / 61067.1 = 3.78x**
  （old 为 3.69x）；
- 3.78x 已落在 1D 应用级 `3.85–9.70x` 的低端附近，而不是 100-step 看起来的
  “2.71x 差很远”。

主要矛盾因此修正为：

1. **未计时 per-step 约 13–16 ms（21–26%）**：任务调度/同步/MPI 等待/输出等
   未进入 StageProfiler 类别的 host 开销；这是当前最大的单一未知块。
2. **interface 约 25.9 ms/step（42% per-step）**：`mpi_interface 12.74` +
   `swap 5.90` + `gradient_mirror/halo 7.26`。
3. reconstruction 7.15、gradient 11.56（其中 kernel 只有 0.86）。

cell-gather 的 stage −8% 只带来 1000-step wall −2.4%，因为第 1 项的未计时
per-step 没有随之下降。1000-step 下 fixed 只占 3%，**不是**主要矛盾；100-step
下 fixed 占 23%，所以 100-step 不适合作为 3D 的生产基线。

**修订预期（1000-step 口径）**：若把“未计时 per-step + interface”合计约
41.7 ms 压掉 20–30%（8–12.5 ms），per-step 61.1 → 48.6–53.1 ms，wall
61.1 → 48.6–53.1 s，CPU/HIP 比值 **3.78x → 约 4.1–4.6x**。这比 100-step 的
“2.8–2.9x”预期更合理，也说明 4x 在 1000-step 口径下是有希望的（中等把握），
在 100-step 口径下则基本不可能。

**基线问题**：目前 129³ 的 31.25x 是“均匀流 box、1 rank、50 steps、单次几何”
的一个点，数据支持不足以当正式基线；至少还需要真实 flow、不同 steps、8-zone
2.1M 和 CPU 多核对照。这里只把它作为“规模趋势”的证据，不写成性能结论。

### 仍未闭合

- **接口紧凑布局**与 **gradient_mirror / reconstruction 去搬运**仍是性能项，不是
  正确性缺口；当前没有把 `SwapBcGrad` 的 host 打包改掉。
- **多卡 8:1 核卡比**仍缺合法的 ≥64-zone fixture（`simutask=Partition` 未实现，
  但 `simutask=Grid`/`gridObj=3` 可生成分区网格；缺的是 ownership/interface manifest）。
- **limiter≠0、multigrid、LUSGS、viscous/RANS** 仍未迁移，host 回退；因此不能把
  本轮结果写成“完整 NS 求解器已 HIP 化”。

## 2026-10-21 resident q generation contract, multi-zone device gradient + reconstruction

**目标**：把多 zone HIP 从“只验证 fused flux/interface”推进到“device gradient 与
device reconstruction 消费 state-owned device q”，并且用可验证的 generation
contract 替代隐式假设。没有新增 standalone port，也没有放宽 tolerance。

**根因（P0）**：device `cellState` 与 host `q` 镜像没有任何同步契约。host RK
state update、host `NsCalcBc`、host interface exchange 都会改写 host `q`，device
完全看不到；而 device gradient 的 resident 判定只看 owner generation，所以多 zone
HIP gradient 会读到过期 owned/ghost 值。这正是之前自动打开 gradient 时 qf/state
出现 ~9e-3、boundary 自动路径出现 ~4.6e-3 的来源，而不是 kernel 精度问题。

**本轮实现**

- 新增 backend-neutral `codes/accel/include/ResidentStateContract.h`（header-only,
  可在无 DCU 的 CPU 上测试）：绑定 owner token/topology generation/extents/方程数，
  区分“owned cell 整场 generation”与“ghost epoch”，提供
  `RequiredSync()`/`NoteFullFieldSync()`/`NoteGhostSync()`/`NoteHostGhostMutation()`/
  `NoteDeviceFieldUpdate()`/`NoteDeviceGhostUpdate()`/`Invalidate()`。
- `HipGradientStorage` 以该 contract 作为 resident q 的唯一语义来源，删除
  `residentStateReady`/`cachedStateOwnerToken`/`cachedStateCell*` 一套平行状态。
  gradient、local timestep、boundary、primitive update、boundary pack/unpack 全部
  走同一套 sync 决策。
- solver 侧显式发布 host 写入：`UNsUpdate` 的 host RK 更新
  (`NotifyHostFieldStateModified` → owner generation +1)、`NsCalcBoundary` 的 host
  ghost 写入（只刷新 ghost，`O(nGhostCells)` H2D）、interface target ghost 写入
  （每个 exchange 配对一次 device unpack 清除，未覆盖的由轮末安全网补齐）。
- 多 zone device gradient 重新打开的条件从“必须显式 opt-in”改为“必须存在 resident
  halo contract（HIP interface）”，并新增 trace 可验证输出
  `RESIDENT_STATE_CONTRACT ... mismatched_values/max_abs_diff`。
- 多 zone device reconstruction：`CellFaceReconstructionView` 增加
  `syncGradientGhostsFromHost`；跨 zone 时 gradient 仍做 D2H + `Grad::SwapBcGrad`
  （把 interface ghost gradient 的权威来源显式化），只把 ghost 切片（3 分量/方程）
  上传回 device，然后 reconstruction/flux/residual 全程 device。
- `UNsGrad::CalcGradHip` 的下载决策改成 capability 参数，
  `UseHipDeviceReconstruction()/UseHipGradient()` 对多 zone 统一要求 interface
  contract，不再用环境变量猜测。
- 新增 `tests/hip_resident_state_contract_test.cpp`（12 例）与
  `oneflow_hip_resident_state_contract_test` target；root HIP build 现在会补装
  `bin/system`，避免 fresh build 无法启动 case；multi-zone trace runner 只为 HIP
  腿设置 `ONEFLOW_ACCEL_BACKEND=HIP`，并断言每个 rank 有非零 `HIP kernel` 阶段，
  防止 CPU fallback 被当作 HIP 证据。

**Kunshan 证据（revision `f0af059f`，DTK 26.04 / gfx906 / dcu:1）**

- root HIP build + DCU preflight 通过；硬件 CTest `16/16`；HIP GoogleTest `15/15`；
  `oneflow_hip_resident_state_contract_test` `12/12`。
- 8-zone fixture gate：`MULTIZONE_FIXTURE_PASS ranks=8 zones=8 interfaces=35624`。
- 3-step trace（8 ranks，`ONEFLOW_ACCEL_BACKEND=HIP` + reconstruction opt-in）：
  `MULTIZONE_TRACE_PASS`，每 rank `STAGE_TRACE_PASS records=36`，
  HIP vs legacy 最大绝对差 `2.2726265314076954e-13`（cpu_batch 为 0）。
- resident contract 检查：`records=24 failures=0 max_abs_diff=0`（8 ranks × 3 steps，
  owned+ghost 全量逐值比较）。
- fixed-CFL `0.01`、50-step stability：legacy / cpu_batch / hip_batch 三路
  `exit_code=0`、`diagnostic_hits=0`、`PASS`；每路 8 ranks 均有 solver work。
- 同 basis paired timing（`steps=10,warmup=1,repeats=3`，8 zones、8 CPU ranks vs
  1 DCU/8 ranks，fixed CFL `0.01`）：CPU mean `4419.345 ms`，HIP mean
  `2924.047 ms`，`CPU/HIP = 1.51153x`。同 revision 的 50-step stability solver
  时间（每 rank，含影响口径差异）为 legacy `20.2–22.4 s`、HIP `7.3–8.4 s`。
- 50-step stage breakdown（rank 0，HIP vs legacy，ms）：gradient `1545.0 / 4408.8`、
  reconstruction `564.1 / 2845.8`、update_residuals `3044.8 / 10354.2`、
  mpi_interface `555.6 / 888.2`、HIP kernel `46.0 / 0`。
- 与之前记录的 `553a9cc5b`（`steps=10,warmup=1,repeats=3`，CPU `4275.308` /
  HIP `4269.301`，`1.0014x`）相比，本轮是同一 basis 的新测量；节点不同，不能把
  两次数据混算，但方向从“持平”变为“HIP 明显更快”。

## 2026-10-22 性能切片：接口 pack 去重 + aero 力直接读 cell gradient

**热点确认（100-step，8 ranks 全 rank 聚合，非只看 rank 0）**

`update_residuals` 31.4 ms/step（父项）、`mpi_interface` 11.7、`gradient` 19.2
（子项：kernel 4.1、mirror 3.7、其余 ~11.4 = `SwapBcGrad`）。细分后
`swap_store`（`Grad::StoreBcGrad`）**4.1 ms/step** 是最大单块，`swap_upload`
2.3、`swap_download` 1.6。

**切片 1（保留，`c15324dc`）**：`UploadInterfaceValue`/`DownloadInterfaceValue`
每次都新建 endpoint 向量 + `nEqu×nIFaces` packed 缓冲，再二次拷贝进/出接口
storage，并单独做一遍 endpoint 校验。接口 storage 本身就是 equation-major 的
逐面布局，改为直接 copy/scatter，endpoint id 每 ghost level 建一次（thread-local
scratch），范围检查并入 id 遍历。值、面序、交换时序不变。

- 交错配对（同节点、100 steps、3 对）：old `9770.9 ms` → new `9436.9 ms`，
  mean 比值 **1.0354x**，3/3 对为正。
- 正确性：CPU normal/strict 各 5/5；3-step trace PASS；50-step stability 3/3；
  NoTrace 200-step residual 相对差 1.094e-14。

**切片 2（保留，`95830d9b`）**：`StoreBcGrad` 每个 RK stage 用随机 gather 把
`dqdx/dqdy/dqdz` 在边界面左单元的值抄到 `bcdqdx/y/z`（3×neq×nBFaces），
唯一消费者是 `AeroForceTask`。改为让 force task 直接读 `dqdx/y/z[eq][leftCell]`
（同一数组、同一单元、数值逐位相同），并默认关闭这次拷贝（
`ONEFLOW_ENABLE_BC_GRADIENT_STORE=1` 可恢复）。

- **中间失败**：只关拷贝、不改消费者时，CPU 五算例 normal/strict 各 **0/5**
  （`aero.dat`/`wallaero.dat` 变化）——说明 `bcdq*` 确实是 force 的输入，
  直接删除是错的。改消费者后恢复。
- 最终交错配对（`500ba688` → `95830d9b`，同节点、100 steps、3 对、每对 2 样本）：
  - old 原始样本：10348.98 / 9568.04 / 9808.82 / 9621.76 / 9694.39 / 9672.34 ms
  - new 原始样本：9038.43 / 8976.67 / 8901.26 / 9207.59 / 9106.78 / 9484.20 ms
  - mean `9785.7 → 9119.2 ms`（**1.0731x**）、median `9683.4 → 9072.6 ms`（1.0673x）
  - 配对均值比值：1.1056 / 1.0730 / 1.0417，3/3 对为正
  - 同作业 rank 0/4/7 的 loop 求和：`57.95 → 52.44 ms/step`（**−9.5%**）；
    `gradient 18.38 → 12.68`，其中 `swap_store 4.12 → 0`
- 正确性（`95830d9b`）：CPU normal/strict 各 5/5；8-zone 3-step trace PASS +
  resident contract `24/0/0`；8-zone 50-step stability 3/3 PASS；
  NoTrace 200-step residual 相对差 **1.094e-14**。

**下一最大机会**：`swap_upload + swap_download` 仍约 4.3 ms/step，是
`SwapBcGrad` 为接口 storage 做的一次随机 gather/scatter；若要继续压缩，需要让
接口交换直接消费 endpoint 紧凑布局（而不是逐面 `MRField`），属于通信布局改动，
需单独立项。`gradient_mirror 3.5` 与 `reconstruction 9.0` 次之。

## 2026-10-22 闭环审计与残差/正性诊断切片

### 审计结论（当前受限路径：inviscid / Lax-Friedrichs / limiter-off / 显式 RK）

| 字段 | 权威位置 | host 镜像 | 谁在写 | 谁在读 |
|---|---|---|---|---|
| `q`（owned+ghost） | device `cellState` | 每 stage D2H（state update 回写）→ **当前** | `UpdatePrimitiveState` + `UpdateBoundaryState` + ghost sync | 输出/气动力/restart 写盘 |
| gradient | device `cellGradient` | 仅 boundary-adjacent owned + ghost（`SwapBcGrad` 需要） | `CalcGradient` + `MirrorGradientCellsToHost` | 仅 host 梯度交换 |
| residual | device `storage.residual` | **不再每步镜像**；trace/输出/restart dump 时才下传 | `CalcAndAddReconstructedFaceFlux` / `ScaleCurrentResidual` | 本轮改为 device 归约标量 → `ResidualTask` |
| timeStep / invsr | device | 无（`downloadToHost=false`） | `CalcLocalInviscidTimeStep` | `ScaleCurrentResidual`（`preferResidentTimeStep`） |
| `bc_q`、boundary ghost | device | 每 stage D2H（ghost + bcQ） | `UpdateBoundaryState` | `NsCalcGamaT`/输出 |

- 真正必要的 host 读取：输出、restart、气动力（`wallaero`/`aero`）、残差报表。
- 旧 task 接口造成的整场镜像：residual 每步整场（本轮删除，改标量）、gradient 整场（上一轮删除，改 boundary 邻域 + ghost）。
- 同步点：`RefreshResidentPrimitiveState`（full/ghost/None，由 generation+ghost epoch 决定）、`NoteHostGhostMutation`（host 边界/交换后）、`NoteDeviceFieldUpdate`（state update 后）、`NoteDeviceGhostUpdate`（device 边界/unpack 后）。
- state owner / generation / halo contract 可以直接复用，无需新建。

**主要缺口（按对“可用求解模式”的影响排序）**

1. **残差诊断曾是假的**：device 路径不回写 host residual，`res.dat` 每步都是 0，收敛历史、restart 里的 residual history、host 正性守门全部失效（trace 门禁看不出来，因为 trace 会把 residual 下传）。
2. restart 等价性没有端到端证据（q 是当前的，理论应等价，但没有 100 步 = 50+restart+50 的实测）。
3. 6 个 `boundary_*` 子项与 `update_residuals`（父项）存在嵌套；`boundary_device` 之前没注册。
4. 多卡/跨节点 8:1 仍需 ≥64-zone fixture。

### 本轮切片（已实施，`80b0dec3` → `500ba688`）

`ComputeResidualMetrics`：在 device 上把**已缩放** residual 归约成 host task 需要的标量
（每方程 Σres²、max|res| 及 cell、primitive 最小 rho/p），复用原 `ResData`/MPI 平均
路径，只回传约 12 KB/zone；residual 整场只在 trace / 输出 / restart dump 时下传。
同一归约顺带让 `ONEFLOW_FAIL_ON_NONPHYSICAL` 对 device 路径真正生效（以前依赖被跳过的
host cell loop）。期间该守门抓到自己一个 bug：q 分量 stride 应用 owned+ghost
（`500ba688` 修正），否则假报非物理。

**验收（`500ba688`，8-zone / 8 ranks / 1 DCU，backend-only）**

| 项 | 结果 |
|---|---|
| 200-step residual history vs CPU（高精度输出） | `max_rel_diff = 1.068e-14`（200×5 全部行） |
| 200-step `aero.dat` vs CPU | 5 位有效数字内完全一致 |
| 3-step trace + contract | PASS，36 records/rank，`failures=0 max_abs_diff=0` |
| 8-zone / 单 zone 50-step stability | 三路全 PASS，`hits=0` |
| `residual_metrics` 成本 | 0.77 ms/step（整场 D2H 约 10 ms/step 的 1/13） |
| 100-step HIP | 9321 / 10156 ms，与上一轮持平（无回归） |

**下一片（已明确，未实施）**：restart 等价性。步骤：`nFieldSave=50` 跑 50 步产生
restart → 用 `startStrategy` 重启再跑 50 步 → 与连续的 100 步run 比较最终
`res.dat`/`aero.dat`/restart 内容；CPU 与 HIP 各一组。判据：residual 相对差
≤1e-12、aero ≤1e-10、`exit_code=0`。同时清理 stability/timing runner 的继承环境
污染（本轮已改为 backend-only 并记录 auto/1/0）。

## 2026-10-22 默认 policy 收口：只设 `ONEFLOW_ACCEL_BACKEND=HIP`

目标要求“通过 `ONEFLOW_ACCEL_BACKEND=HIP` 切换 GPU 后段”，而此前 reconstruction /
state update / boundary / resident time-step 都要显式变量才开。`63793e16` 把已验证的
阶段全部挂到 policy，随后用**纯 backend-only** 门禁反过来抓到两个遗留 bug：

1. `NsCalcBoundary` 仍只认 `ONEFLOW_ENABLE_UNS_HIP_BOUNDARY` 这个显式变量 →
   policy-only 运行时静默回退 host `NsCalcBc`（100-step 里 `boundary_bc≈1018 ms`）。
2. `ULhs::UseResidentHipTimeStep()` 只读显式变量 → policy-only 时 `CALC_LHS`
   把**过期的 host timestep** 上传去缩放残差，导致 state 差 8e-3。

`bd35bac9` / `03cae1bb` 修掉：两者都改走 `HipProductionFeatureEnabled()`，显式请求
仍保留 fail-fast。这也说明 backend-only 门禁不能省——之前的显式变量门禁掩盖了它们。

**`03cae1bb` 门禁（全部 backend-only，不带任何 `ONEFLOW_ENABLE_UNS_HIP_*`）**

| 门禁 | 结果 |
|---|---|
| 8-zone 3-step trace | `MULTIZONE_TRACE_PASS`，每 rank 36 records |
| resident contract | `records=24 failures=0 max_abs_diff=0` |
| 8-zone 50-step stability | legacy / cpu_batch / hip_batch 全 `exit_code=0`、`hits=0`、PASS |
| 单 zone 3-step stage trace | `STAGE_TRACE_PASS records=36` |
| 单 zone 50-step stability | 三路全 PASS（之前的 FAIL 是本轮脚本把 8-rank manifest 泄漏到 1-rank 腿，已修） |
| 硬件 CTest / contract | 16/16，resident-state contract 12/12 |

**端到端（100 steps，全开默认）**

| 对照 | 样本 | mean | 比值 |
|---|---|---:|---:|
| 8C（`kshcnormal`，无卡） | 22889.8 / 23658.8 / 24468.7 / 24526.1 | 23885.9 ms | **2.50x** |
| 8C + 1 DCU（`kshdnormal`，8 ranks/DCU） | 9332.3 / 9451.6 / 9822.5 / 9549.3 | 9538.9 ms | — |
| 同节点 8C（DCU 节点） vs 8C+1DCU | 28657.9 / 30517.1 vs 9212.1 / 10707.7 | 29587.5 / 9959.9 | 2.97x |

- HIP 侧稳定（9.3–9.8 s，±3%）；**CPU 基线才是噪声源**（CPU 队列 21.9–24.5 s，
  跨节点 ±5%），所以比值区间 2.30–2.50x，引用时必须带样本。
- 资源标注：CPU 腿 8 核；HIP 腿 8 核 + 1 DCU（8 ranks/DCU = 规范 8:1 槽）。

**仍未闭合**

- 多卡 8:1 核卡比：缺 ≥64-zone fixture（`Partition` 任务未实现，需外部 METIS 分区）。
- DeviceDirect：probe 通过但实测慢 1.59x，MPI 内部仍 host staging。
- `SwapBcGrad` 的 host 打包（~10 ms/step）与未计时 ~19 ms/step。
- 跨节点 8:1 核卡比（受同一 fixture 与 7+1 分布限制）。

## 2026-10-22 跨节点打通与 fixture 扩展评估

### 跨节点阻塞已定位并修复

之前 2 节点作业的段错误不在 OneFLOW：backtrace 是
`mca_coll_han_comm_create → mca_coll_han_bcast_intra → MPI_Bcast`，即 OpenMPI 的
`coll/han` 分层集合通信在 **7+1 的不均匀分布**（`-N 2 -n 8` 默认 block 分布）上
初始化崩溃，发生在 `SimuContext::SetupCaseEnvironment()` 的 Bcast 里。

- 修复：`OMPI_MCA_coll='^hcoll,han'`（CPU 队列 2 节点 8+8 分布下 `han` 本来能跑，
  所以这是“不均匀分布 + han”的组合问题；直接排除该组件最稳）。
- 证据记录：timing runner 现在把 `map_by`、`OMPI_MCA_coll`、
  `SLURM_CPUS_PER_TASK`、`SLURM_JOB_CPUS_PER_NODE` 写进 `environment.txt`；
  `ONEFLOW_MZ_MAP_BY` 可指定 `ppr:N:node`（本例 Slurm 实际分布仍是 7+1，
  portal 不接受 `--ntasks-per-node`，所以最终按 7+1 跑，仍是真实跨节点 halo）。

**跨节点结果（revision `ed1e7d1e` 脚本 + `7ab873b8` 二进制，2 nodes，8 ranks，2 DCUs=1/节点）**

- 3-step trace：`MULTIZONE_TRACE_PASS`，8/8 rank 有 solver work，
  contract `records=24 failures=0 max_abs_diff=0`。
- 100-step：CPU `27970.2/28661.8 ms`（mean 28316.0），HIP `11106.2/11122.0 ms`
  （mean 11114.1）→ 2.548x。
- **资源标注**：8 核 / 2 DCUs = **4 核/DCU**，不是 8:1 槽；加上 7+1 分布，
  这条只能记为“跨节点正确性 + 跨节点可用性”，**不能用于核卡比**。

### fixture 扩展评估：仓库内做不了

`simutask=Partition` 在当前构建里是 `throw "Task \"Partition\" is registered but
not implemented in this build."`（`codes/main/src/SimuTaskReg.cpp`）。现成 fixture
只有 `m6wing-4.ofl` / `m6wing-8.ofl`。因此：

- 8:1 规范的多卡配置（N DCUs × 8 核、1 rank/zone）需要 ≥N 个 zone；要 8 卡就要
  ≥8 zone（有）**但** CPU 参照需要 ≥64 ranks，即 ≥64 zones；
- 64-zone 分区必须由外部 METIS 分区流程或 fixture owner 提供，当前 build 无法生成。

结论：多卡核卡比在拿到 ≥64-zone fixture 之前无法按规范给出；跨节点正确性已闭合，
跨节点 8:1 核卡比同样需要该 fixture（或让 7+1 分布变成 4+4 并给 2 卡配 16 核）。

## 2026-10-22 资源口径纠正：1 DCU 槽 = 8 核，CPU 基线必须在 CPU 队列

昆山活配置（`scnet.py queues --details`）：`kshdnormal` 32 cores/node、**4 DCUs/node**
→ **一个 DCU 槽 = 8 核**；`kshcnormal` 32 cores/node、**0 DCUs**。之前几处口径不严谨：

1. HIP 腿写“1 DCU”，实际是 **8 核 + 1 DCU**（8 ranks × 1 core，8 ranks/DCU）。
   按 8:1 槽约定，1 DCU 配 8 核是对的，但必须写成“8C+1DCU”，不能只写“1 DCU”。
2. CPU 基线之前跑在 **DCU 节点**（同一 allocation 里挂着空闲卡），不是 CPU 队列。
   实测同样 8 核：`kshcnormal` **20854.4 ms** vs `kshdnormal` **28558.5 ms**（+37%）。
   按规范 CPU-only 作业必须用 `kshcnormal`。
3. 之前报告的 4 DCU 点（8 ranks / 4 DCUs）= **2 核/DCU**，不是 8:1 槽，
   只能算“增卡实验”，不能用于核卡比或加速比。矩阵脚本现在会拒绝非 8 核/DCU
   的腿（除非显式 `ONEFLOW_MZ_MATRIX_ALLOW_NONSTANDARD=1`），并记录
   `cores`、`cores_per_dcu`、`ranks_per_dcu`。

**100-step、全开阶段、修正后的两组数（revision `7ab873b8`）**

| 对照 | 资源（CPU 侧是真实活跃核） | CPU mean | HIP mean | 比值 | 备注 |
|---|---|---:|---:|---:|---|
| 规范资源口径 | 8C（`kshcnormal`，无卡） vs 8C+1DCU（`kshdnormal`，8 ranks/DCU） | 20854.4 ms | 9521.9 ms | **2.19x** | CPU 基线在 CPU 队列，跨节点型 |
| 同节点隔离 | 8C（DCU 节点，卡空闲） vs 8C+1DCU | 28558.5 ms | 9521.9 ms | **3.00x** | 同节点，隔离“加卡”效果 |
| 参考（无效比值） | 8C vs 8C+4DCU | 28558.5 ms | 2347.0 ms（10-step 旧点） | — | 2 核/DCU，非 8:1，不可用于核卡比 |

HIP 4 个正式 sample：9536.8 / 9425.7 / 9729.6 / 9395.4 ms（同节点两次作业）。

结论：

- **规范口径下 1 个 DCU 槽（8 核）对 8 核是 2.19x**；同节点隔离“加卡”是 **3.00x**。
  两者都要报，且必须写清 CPU 腿在哪个队列/节点。
- CPU 腿对节点类型敏感（37%），这个量级和部分“加速比”同阶，所以任何跨队列
  比值都必须标注节点差异，不能与同节点比值混用。
- 多卡 8:1 槽要求 `cores = 8 × dcus`：8 卡需要 64 核 + 至少 8 个 zone 的 fixture。
  当前 8-zone fixture 只有 8 个 rank，做不到 8 卡 × 8 核的规范配置，需要新的
  分区 fixture；在那之前不声明多卡核卡比。

## 2026-10-22 交换值评估、mirror-only 梯度、固定开销与跨节点阻塞

### 交换值评估（并行正确性）

跨 zone 的 ghost 梯度必须与 CPU oracle 逐位对齐，所以先确认“交换什么、什么时候交换”：

- **交换的值**：每个 interface face 上，发送侧 *owned (left) cell* 的 3 个梯度分量
  × 5 方程。接收侧写入对应的 *ghost (right) cell*。这与
  `UploadInterfaceValue` 的 `GetSId(iFace, ghostId+1)` / `DownloadInterfaceValue`
  的 `GetTId(...)` 完全一致，也和 q halo 用同一套 endpoint cell id。
- **时机（关键）**：legacy 的 `Grad::SwapBcGrad` 只把 owned 值写进 send storage，
  真正的收发发生在之后的 `CUpdateInterface::Run()`；因此第 k 步重建用的 ghost
  梯度其实是邻居第 k−1 轮交换的值（**一轮滞后**）。同一 stage 内的“pack→MPI→unpack”
  会变成同轮值，qf2 会偏离 oracle（量级 1e-2），trace 必然失败。
- **物理边界**：`GradientCopyBoundaryKernel` 的 left→ghost 拷贝对应
  `CalcGradGGCellWeight` 的同一拷贝，不能被 halo 覆盖。

结论：**不改交换值和时序**。只把 host 侧的“整场镜像”缩小到 host 交换真正读的那一小块：
interface send 用的 boundary-adjacent owned cell（`dqdx[leftCell]`，`StoreBcGrad`
用的也是同一组），数量 ≤ `nBFaces`。host mirror 仍然喂给未改动的
`Grad::SwapBcGrad`（保留一轮滞后），ghost 切片仍然回传 device 供重建使用。

实测（`743bc421` → `7ab873b8`，100 steps，rank 0）：

| 项 | 之前 | 之后 |
|---|---:|---:|
| gradient 总计 | 2789.8 ms | **1647.7 ms** |
| 其中整场下载 / mirror | 1078.0 | **290.6**（mirror，含 pack） |
| gradient kernel | 365.5 | 383.1 |

门禁：3-step trace `MULTIZONE_TRACE_PASS`（36 records/rank，HIP vs legacy
2.27e-13），contract `records=24 failures=0 max_abs_diff=0`，50-step stability
三路 `exit_code=0 / diagnostic_hits=0 / PASS`。

### 端到端（100 steps，单节点 8 核 vs 1 DCU，8 zones / 8 ranks）

| 腿 | formal 1 | formal 2 | mean |
|---|---:|---:|---:|
| CPU 8 cores | 28083.6 | 27630.7 | **27857.2 ms** |
| HIP 1 DCU | 9444.4 | 9574.1 | **9509.3 ms** |
| 比值 | | | **2.930x** |

（同一 revision 之前为 2.556x；本轮 mirror 改动带来约 +15%。）

### 固定开销 profile（同一节点，cases=1/2/100 steps，HIP）

| steps | wall（warm） |
|---:|---:|
| 1 | 1940.7 ms |
| 2 | 1993.6 ms |
| 100 | 9275.3 ms |

线性拟合：**固定开销 ≈ 1.87 s/run，稳态 ≈ 74 ms/step**（100-step 时固定部分占
wall 的 20%）。逐 stage 求和约 55 ms/step，说明还有约 19 ms/step 未被现有计时覆盖
（task 调度、postprocess、残差/CFL 输出、收尾写盘）。

### 跨节点：当前阻塞在一个与 HIP 无关的 setup 崩溃

`--nodes 2 --cpus 8 --dcus 2`（实际分布 7+1 ranks）下，**legacy CPU 腿**在
`SimuContext::SetupCaseEnvironment()` 段错误（rank 6，`addr2line` 确认：
`SetupCaseEnvironment → SimuImp::RunCase → Simulation::RunImpl → main`）。
它发生在任何 halo/HIP 代码之前，说明 3D fixture 的多节点 case 准备本身还没打通；
`--scheduler-option=--ntasks-per-node=4` 未能改变 7+1 的分布。跨节点 correctness
和跨节点加速比因此仍未取得证据，不能声明。

## 2026-10-21 loop 优化：measurement basis、limiter、device time-step、pinned mirror

**先纠正口径**：`steps=10` 时每次 mpirun 有约 3.8–4.1 s 固定开销（MPI/设备初始化、
8-zone 网格读入、写盘），HIP 的 loop 只有 ~0.8 s，所以 10-step 的“端到端比值”
主要测的是启动。改按 100 steps + 逐 stage 求和两种口径同时报。

**revision `743bc421`（Kunshan，8 zones / 8 ranks / 1 DCU，fixed CFL 0.01）**

| 口径 | CPU 8 cores | HIP 1 DCU | 比值 |
|---|---:|---:|---:|
| 100-step wall（`steps=100,warmup=1,repeats=2`） | 28534.8 ms | 11166.7 ms | **2.556x** |
| 同 revision 的 loop 求和（rank 0，per step） | ~253 ms/step | ~62 ms/step | ~4.1x |

**本轮实际改动**

- `LimField::Init` 复用 `qf1/qf2`（原每 stage 重新分配 ~9 MB）：limiter
  1147→135 ms/100 steps。
- 多 zone device local time-step 打开（读 resident q，结果留在 device 供 CALC_LHS）：
  `calc_time_step` 581→82 ms/100 steps。
- 跨 zone 梯度镜像改走 pinned staging（`hipHostMalloc`，一次大拷贝 + host scatter）：
  `gradient_download` 2734→1078 ms/100 steps（~200 MB/s → ~500 MB/s）。
- 新增 `gradient_download`/`gradient_kernel`/`halo_ghost_sync`/`state_update` 分项计时。
- 尝试过“HostStaged 下去掉重复 halo 传输”，但 contract 检查显示
  `owned_mismatches=0, ghost_mismatches>0`（最大 1.6e-2），说明配对的 device
  unpack 才是让 device ghost 与 host 一致的那一步；已回退，去重需要真正的
  device-side halo（双缓冲 + 保持 legacy 一轮滞后语义）。

**当前 100-step、rank 0 的分项（ms/step）**

| 项 | ms/step | 说明 |
|---|---:|---|
| gradient（含 download 10.8 + kernel 3.7） | 27.9 | 跨 zone ghost 梯度需整场 D2H |
| mpi_interface | 15.6 | host exchange + device pack/unpack |
| reconstruction | 9.1 | device kernel + ghost 梯度上传 |
| update_flowfield | 2.5 | host `tempr` 重算 |
| state_update | 2.1 | device kernel + host mirror |
| calc_boundary | 2.0 | device kernel + host gamma |
| load_residuals / load_q / calc_lhs | 3.1 | host 桥接 |
| limiter | 1.3 | 只剩 5×nTCell 填充 |
| HIP kernel（flux） | 1.0 | — |
| 固定开销（wall − loop） | ~3.8 s/run | MPI/grid/HIP init + 写盘 |

**下一步最有价值的两个方向**

1. 把跨 zone 梯度从“整场 D2H”改成“只交换 interface 值”：device 端 pack
   interface-left 梯度（~0.5 MB）→ host MPI → unpack 到 device ghost，并用双缓冲
   复现 legacy 的一轮滞后语义。预计 gradient 27.9 → ~6 ms/step。
2. 去掉 host exchange 与 device pack/unpack 的重复（15.6 ms/step），前提是 1 完成，
   让 device halo 成为唯一权威。

## 2026-10-21 多 zone device state-update / boundary 打开与扩展性矩阵

**代码（revision `6759383e`）**

- `NsCalcBoundary` 允许 device boundary 进入分区算例：kernel 本来就跳过
  `INTERFACE(-1)`、`PERIODIC(99)`、`OVERSET(1000)`，因此 interface ghost 仍归
  interface contract，物理边界 ghost 在 device 上算。修掉 gate 里的 stale
  `ug`（`NsCalcBoundary` 每 zone 调用，缓存的 unstructured view 可能仍指向上一个
  zone），并新增 `HIP_BOUNDARY_DIAGNOSTIC` 一次性拒绝原因，不再静默 false。
- `Limiter::CalcLimiter` 在 `ILMT_NO`（`ilim=0`）下直接填充 `phi=1` 并返回，
  不再每个方程 allocate/free `minvf/maxvf`。数值完全一致（无算术），CPU normal/
  strict 五算例各 5/5 复验通过。
- 新增 `ci/kunshan/f2-main-solver-multizone-matrix.sh`：同一 8-zone case 上按
  CPU ranks × visible DCUs 跑配对 timing，逐腿检查 exit code，输出
  `matrix.tsv` / `matrix-ratio.tsv`。
- multi-zone trace runner 增加 `ONEFLOW_MZ_HIP_BOUNDARY`/`ONEFLOW_MZ_HIP_TIME_STEP`
  开关，便于逐阶段二分。

**阶段门禁（`6759383e`，8 ranks，3 steps trace + 50 steps stability）**

| 阶段组合 | 3-step trace | 50-step stability |
|---|---|---|
| recon | PASS（36 records） | legacy/cpu/hip 全 PASS，hits 0 |
| recon + state update | PASS | 全 PASS，hits 0 |
| recon + boundary | PASS | 全 PASS，hits 0 |
| recon + state update + boundary | PASS | 全 PASS，hits 0 |

resident contract 在每次 gradient 前比较 owned+ghost，全部 `failures=0`、
`max_abs_diff=0`。

**同 basis 扩展性矩阵（`steps=10,warmup=1,repeats=3`，8-zone、8 ranks、fixed CFL 0.01）**

| ranks | DCUs | ranks/DCU | CPU mean ms | HIP mean ms | CPU/HIP |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 8 | 3458.459 | 2566.409 | 1.3476x |
| 8 | 4 | 2 | 3458.459 | 2347.003 | 1.4736x |

- 4 DCU leg 的 solver 日志报告 `OneFLOW HIP device: 0/4`，rank→device 绑定是
  `localRank % deviceCount`，即 2 ranks/DCU。
- 同一份 8-zone fixture 只能配 8 个 rank（zone→rank 映射固化在 fixture 里），
  因此本轮不能给出 4 ranks（2 zones/rank）或 16/32 CPU cores 的对照；需要新的
  分区 fixture 或重新生成 ownerRanks 才能扩 CPU 侧口径。
- 与 `f0af059f` 的 `1.51153x` 不是同一节点，CPU leg 本身从 4419 ms 变到
  3458 ms，两个比值不能混算。

**50-step 阶段级变化（rank 0，`ms`；recon-only → +state+update+boundary）**

| stage | legacy | recon-only | 全开 |
|---|---:|---:|---:|
| gradient | 4408.8 | 1545.0 | ~790（10-step 口径 158） |
| limiter | 601.8 | 683.7 | ~284（fast path） |
| reconstruction | 2845.8 | 564.1 | ~357 |
| update_residuals | 10354.2 | 3044.8 | ~1560 |
| mpi_interface | 888.2 | 555.6 | ~503 |
| calc_boundary | ~520 | 522.8 | ~100 |
| calc_lhs | — | 73.9 | ~31 |
| H2D/D2H | 0 | 0 | 0 |

> 注：上表“全开”列由 10-step 正式 repeat 的 rank-0 分项按比例给出量级，只用于
> 说明方向；精确的 50-step 全开分项要在下一次 stability 作业里重新汇总。

**GPU-aware MPI / DeviceDirect（重要负面结论）**

- 2 节点设备指针 probe 通过：`GPU_AWARE_MPI_TWO_NODE_PROBE_PASS`，
  `gfx906`，device pointer `MPI_Sendrecv` 收发的数值正确。
- 但同一 revision、同一 basis 打开 `ONEFLOW_ENABLE_UNS_GPU_AWARE_MPI=1`
  + `ONEFLOW_GPU_AWARE_MPI_CONFIRMED=1`（DeviceDirect）后：HostStaged 三次
  正式 repeat 为 `2688.9/2682.7/2736.1 ms`，DeviceDirect 为
  `3991.5/4410.3/4500.8 ms`，即 **DeviceDirect 慢约 1.59x**。
- 结论：probe PASS 只证明“设备指针能正确传输”，不证明零拷贝。当前
  DTK/OpenMPI 组合下 MPI 实际仍走 host staging，DeviceDirect 必须保持关闭；
  gate 需要追加“DeviceDirect 不得慢于 HostStaged”的 timing 判据。

**仍未闭合**

- reconstruction / state update / boundary 仍为显式 opt-in，尚未进入默认 policy；
  resident time-step 仍只支持单 zone。
- DeviceDirect 在当前工具链下不可用（见上），halo 仍是 host-staged。
- 8 ranks / 1–4 DCU 已测；16/32 CPU cores、4 ranks（2 zones/rank）、跨节点
  与多节点 correctness 仍需要新的分区 fixture。
- 正式性能报告未更新。

**仍未闭合**

- reconstruction 仍为显式 opt-in（`ONEFLOW_ENABLE_UNS_HIP_RECONSTRUCTION=1`），
  尚未加入默认 policy；state update / boundary / resident time-step 仍未在多 zone
  打开，gradient D2H + ghost 上传与 host RK update 的整场 q H2D 仍是每步成本。
- `mpi_interface` 在 HIP 腿包含一次与 host exchange 并行的 device endpoint
  pack/unpack；DeviceDirect 仍要求 GPU-aware MPI probe，未默认启用。
- 跨节点/多 DCU correctness、H100 之外的多卡扩展、正式性能报告均未更新。

## 2026-10-20 runtime ownership exporter

- `8de93bbe` 增加 `ONEFLOW_MZ_FIXTURE_EXPORT_DIR` runtime exporter。每个 MPI
  rank 在第一次 interface-update 时从实际 `InterFace`/`FaceTopo` 写出
  `rank-<rank>.json` shard，包含真实 send/receive face/cell endpoint、
  zone/rank ownership、owned/ghost cell capacity、equation count 和 memory
  space；不再需要根据 `nBFaces`/`nIFaces` 猜测 endpoint ID。
- `ci/kunshan/merge-multizone-fixture.py` 将所有 rank shard 合并为
  `oneflow.multizone.fixture/v1` manifest，严格拒绝缺 rank、重复 zone、
  未配对 reciprocal endpoint、角色/方程数/memory space 不一致。合并后仍由
  `verify-multizone-fixture.py` 执行最终 gate。
- 本地 Python verifier/merge tests `8/8`、C++ `InterfaceTask.cpp`
  `-fsyntax-only`、Kunshan shell syntax 和 `git diff --check` 已通过。真实
  Kunshan manifest 尚未导出；下一步在可用 8-rank fixture 上设置
  `ONEFLOW_MZ_FIXTURE_EXPORT_DIR` 跑一次 interface-update，再 merge/verify，
  然后重跑 CPU/HIP trace、fixed-CFL stability 和 paired timing。

## 2026-10-20 ownership gate and HIP boundary-cache validation

- Kunshan root HIP revision `553a9cc5`：DCU preflight/build 成功，硬件 HIP
  CTest `16/16` 通过。此前 `58078e8c` build 暴露
  `HipGradientStorage::cachedBoundaryFaces` 重复声明，已在 `553a9cc5`
  删除并复验；该失败是代码回归，不能归因于节点环境。
- 真实 8-zone fixture 已由 solver runtime exporter 生成 8 个 rank shard，
  merge/verifier 输出 `MULTIZONE_FIXTURE_PASS`：
  `ranks=8 zones=8 interfaces=35624`。每个 interface 为 5 方程，
  `HostStaged`；总单向 equation-major value count `178120`，按 `Real=8`
  计算约 `1,424,960` bytes。zone owned cells 均为 `36864`，ghost cells
  分别为 `7680,7574,8040,6532,7336,7158,8192,6152`。
- 同一 manifest 的 Kunshan 8-rank 3-step CPU/HIP trace 通过：
  `MULTIZONE_TRACE_PASS`，每个 rank 有 `STAGE_TRACE_PASS`，36 条记录，
  HIP 相对 legacy 最大绝对差约 `5.9952e-15`。fixed-CFL `0.01`、50-step
  stability 三路（legacy/cpu_batch/hip_batch）均 `exit_code=0`、
  `diagnostic_hits=0`，scheduler `COMPLETED/0:0`。
- 同一 case、同一资源和 basis（`steps=10,warmup=1,repeats=3`，
  CPU 8 ranks，HIP 8 ranks，1 DCU）paired timing：CPU mean
  `4275.308459 ms`，HIP mean `4269.301077 ms`，CPU/HIP `1.001407x`。
  这说明当前 HostStaged 多-zone端到端基本持平；ownership manifest 本身
  不是直接 wall-clock 加速，收益是把错误 topology/ghost overwrite/MPI
  deadlock 风险变成启动前可审计 gate，并为固定 pack layout、persistent
  device state 和后续 DeviceDirect 提供真实计数。
- P1 boundary metadata cache 已在代码中按 owner token、topology generation、
  host pointer identity 和 face count 复用静态 boundary geometry/topology；
  每个 RK stage 跳过 7 次 H2D（4 个 `Real` 数组 + 3 个 `int` 数组），
  `inflow` 动态字段仍每次上传。该改动已随 root HIP build/CTest 验证，
  但尚未在多轮 paired timing 中单独隔离其收益。
- P0 ownership/interface 证据已闭合；尚未闭合的是 GPU-aware MPI
  `DeviceDirect` 实测、跨节点/多 DCU correctness，以及把 manifest 消费
  到 persistent device halo buffer。继续保持多 zone HIP gradient、
  reconstruction、state-update、boundary、time-step opt-in 关闭。

> 最后更新：2026-10-20（3D 多 zone stability runner 修复与 Kunshan 复核；技术路线与历史测量仍以各节记录日期为准）
> 用途：每轮任务开始前读本文档，结束后更新本文档。让任何人或智能体
> 接手时只读这一份就能继续推进。
>
> 配套入口：[`AGENTS.md`](../../../AGENTS.md)（规则与文档地图）、
> [`ci/kunshan/README.md`](../../../ci/kunshan/README.md)（集群流程与标准套件）。

## 2026-10-19 3D 多 zone 8-rank 对 8-rank paired timing

- 算例为 3D `m6wingroe_sa` 的 8-zone 分区，不是已经完成的 1D Euler
  microbenchmark。每个 zone 由一个 MPI rank 负责，单节点 8 个 zone/8 个 rank。
- 准确性门禁：CPU 使用 `8 CPU cores / 8 CPU ranks`，HIP 使用
  `1 DCU / 8 MPI ranks`，两者均单节点；3 steps、fixed CFL `0.01`。
  CPU/HIP 均 `active_ranks=8`，8 个 rank 均写出 breakdown；36 条逐 rank
  stage trace 通过，HIP 相对 CPU 最大绝对差约 `8.53e-14`。
- 2026-10-19 在 endpoint topology 修正 revision 上重新完成 Kunshan root
  HIP build、硬件 HIP CTest `16/16`、3-step trace、50-step stability 和
  同口径 paired timing。trace/stability 使用外部已分区 8-zone fixture；该
  fixture 不是 checked-in single-zone `test/cfd/m6wingroe_sa`。
- 当前 paired timing：同一 8-zone 输入、10 steps、warmup `1`、formal
  repeats `5`、fixed CFL `0.01`，端到端 wall-clock（包含 MPI、初始化、host
  packing、H2D/D2H、HIP kernel、interface 和输出）。
  - CPU：`8 CPU cores / 8 CPU ranks`，单节点，mean
    `6249.223629 ms`。
  - HIP：`1 DCU / 8 MPI ranks`，单节点，每张 DCU 8 个 rank，mean
    `5868.614810 ms`。
  - 明确比值：`CPU / HIP = 6249.223629 / 5868.614810 = 1.064855x`。
    等价地，HIP 为 CPU 的 `0.939095x`。这只是当前节点和当前 revision 的
    复核结果，尚不足以替换正式性能报告或宣称稳定应用级加速。
- 旧的 `3473.812429 ms` 对 `4337.233101 ms` 是另一轮节点/运行环境的历史
  测量，不能与本轮数据混合平均或直接比较。
- 之前 `qf1` 约 `3.1e-3` 的失败 trace 来自残留 HIP gradient opt-in；
  多 zone 路径现已显式清理 HIP gradient/reconstruction/state-update/boundary/
  time-step 变量，并在 `UNsInvFlux::UseHipDeviceReconstruction()` 对
  `nZones > 1` 强制回退 host reconstruction。当前多 zone 正确路径是
  CPU gradient + CPU reconstruction + HIP flux/interface。
- 结论：8-rank/8-rank 资源口径、accuracy、stability 和 rank utilization 已闭合；
  当前复核 timing 显示小幅 HIP 优势，但重复测量的节点敏感性仍然明显。下一优化
  重点继续放在多 zone interface/halo、ghost ownership、host-staged 数据搬运和
  跨 RK stage residency；在形成跨节点/多轮稳定 timing 前，不发布正式加速结论。

## 2026-10-19 interface endpoint topology checkpoint

- `7f2c1d7f` 增加多 zone interface endpoint ownership contract：
  source 使用 `OwnedCell`，target 使用 `GhostCell`。
- `5ea98a88` 将 endpoint topology mapping 提取为 backend-neutral
  `BuildInterfaceEndpoint()`，由 `InterfaceTask` 和 HIP/CPU 路径共享。
- `877d1f16` 删除误接入 `OneFLOW_GeometryCore` 的 heavyweight
  `InterfaceEndpointTest` target，避免轻量 contract 触发完整 geometry 静态库重编译；
  轻量 boundary/interface contract 保留在既有 task/parallel 测试边界内。
- 当前仍未打开：
  `ONEFLOW_ENABLE_UNS_HIP_GRADIENT`、
  `ONEFLOW_ENABLE_UNS_HIP_RECONSTRUCTION`、
  `ONEFLOW_ENABLE_UNS_HIP_STATE_UPDATE`、
  `ONEFLOW_ENABLE_UNS_HIP_BOUNDARY`、
  `ONEFLOW_ENABLE_UNS_HIP_TIME_STEP`。
- 下一刀是把 local zone、interface face、owned cell、ghost cell、send/receive
  buffer 和 host/device memory space 写成明确 contract，再在该 contract 上接 HIP
  ghost synchronization；不能先恢复未经验证的 device gradient/reconstruction。

## 2026-10-20 boundary exchange contract checkpoint

- 在 `codes/parallel` 增加 backend-neutral `BoundaryExchangeContract`，统一描述
  source/target zone 与 rank、方程数、owned source endpoint 和 ghost target
  endpoint，并拒绝 self-zone、角色缺失、端点长度不一致等拓扑错误。
- 增加 `ValidateBoundaryBuffer` 与
  `ValidateBoundaryExchangeBuffers`：校验 equation-major value count、非空
  buffer 的 data pointer，以及 `HostStaged`/`DeviceDirect` 对应的
  host/device memory space。空 interface 允许以空 buffer 安全通过。
- HIP interface path 在 MPI count exchange 前执行本地 buffer contract；同 rank
  多 zone 交换额外执行完整 source/target contract。MPI 调度和默认
  host-staged 行为未改变，parallel 层不依赖 HIP 类型。
- 轻量 `BoundaryExchangeContractTest` 已覆盖 owned→ghost 合法交换、空接口、
  错误 memory space 和错误 endpoint role。当前本机没有 `cmake`/CTest，已用
  `g++ -std=c++20` 编译 parallel contract，并对 `InterfaceTask.cpp` 做语法检查；
  尚未替代 Kunshan 的 CPU 8-zone ownership/synchronization、HIP contract、
  3-step trace、50-step stability 或 paired timing 门禁。
- 新增 `CpuOwnershipSynchronizationContractTest` 轻量 target，独立覆盖
  `DistributedDomainPlan` 的 self-neighbor skip/empty topology、跨 rank 双向
  endpoint role、owned→ghost host pack/unpack、空消息 ghost state 不变和非法
  endpoint。新增 8-zone fixture metadata verifier 及其 Python checks；
  trace/timing runner 在设置 `ONEFLOW_MZ_FIXTURE_METADATA` 时会在 MPI 启动前强制
  校验 ranks、zone ownership、endpoint topology、reciprocal pair、equation-major
  count 和 memory space，并写入 `fixture-validation.txt`。未设置时显式记录
  `MULTIZONE_FIXTURE_UNCHECKED`，不把旧私有 fixture 误报为 ownership 验证。
- 下一步：把 8-zone fixture 的 interface face/endpoint map 和 per-rank
  send/receive buffer 元数据接入可复现 runner；先在 CPU 8-zone 上验证空消息、
  self-neighbor skip、ordinary/interface/periodic/solid boundary 的 ghost
  synchronization，再让 HIP path 消费同一 contract。未完成该门禁前继续保持所有
  多 zone HIP gradient/reconstruction opt-in 关闭。

## 2026-10-20 Kunshan validation checkpoint

- 当前 revision 已用完整 `git archive` 同步到 Kunshan 独立 source 目录，没有用本机
  build 代替目标节点证据。
- `kshcnormal` CPU contract 作业通过：`BoundaryExchangeContractTest 8/8`、
  `CpuOwnershipSynchronizationContractTest 6/6`，工具链为 GCC `16.2.0`、
  OpenMPI `5.0.11`、CMake `4.4.3`。
- `kshdnormal` root HIP build 成功，HIP smoke 成功，硬件 HIP CTest
  `16/16` 通过；构建证据使用 DTK `26.04`、`gfx906`、单 DCU。
- 同一 root binary 的 8-rank 3-step CPU 多 zone workload 已生成 8 份
  breakdown/stage 文件；rank verifier 报告 `active_ranks=8` 且每个 rank 有
  meaningful solver work。
- 8-rank HIP trace 尚未形成 PASS：多次分配分别遇到目标节点
  `hipGetDeviceCount: no ROCm-capable device is detected`，以及 DTK
  `amd_comgr` 动态库加载失败。它们是节点/运行时环境失败，不是数值 trace 失败；
  相关作业不得计入 HIP accuracy/stability 证据，也不能据此扩大 capability。
- 本轮 8-zone 私有 fixture 尚未提供与新 schema 对齐的 endpoint metadata，因此
  runner 明确记录 `MULTIZONE_FIXTURE_UNCHECKED`；CPU rank utilization 通过不等同
  于 ownership metadata gate 通过。下一次应在可用 DCU 节点上重跑，并先为真实
  8-zone fixture 补齐 reciprocal endpoint manifest。

## 2026-10-20 follow-up: fixture discovery and DCU preflight

- Kunshan 私有 workspace 中已定位到真实 8-zone case：
  `/public/home/luql/.scnet-hpc/oneflow/work/partition-3df1778b-8-3d/case`。
  运行日志确认 `nTZones = 8`，8 个 rank 各拥有一个 zone；每个 zone 的
  `nBFaces`/`nIFaces` 也已由 solver 读出。该目录和网格文件继续留在集群私有
  workspace，不复制到仓库或公开报告。
- 只读检查没有发现现成的 `ownerRanks`、endpoint cell/face map 或
  `oneflow.multizone.fixture/v1` manifest。已有 8-zone trace 因此仍只能作为
  solver/rank-utilization evidence，不能升级为
  `MULTIZONE_FIXTURE_PASS`；不得根据 `nBFaces`/`nIFaces` 猜测 reciprocal
  endpoint cell IDs。
- 新增 `ci/kunshan/dcu-preflight.sh`，并接入 standalone
  `euler-dcu-gtest.slurm` 与 root `f2-main-solver-build.slurm`。在 configure
  前硬性检查 `rocminfo`、目标 `gfx906` agent 和 `amd_comgr` CMake package，
  输出 `dcu-preflight.txt`（及 rocminfo capture）。无 `/dev/kfd`、错误节点或
  缺 runtime 时作业以 preflight failure 结束，不再伪装成 CMake/build/数值
  failure；HIP 专用流程仍记录为 DTK 例外。
- 当前可并行推进项：由 fixture owner 导出 schema v1 manifest；由 Kunshan
  调度在 preflight 通过的 DCU 节点上重跑 3-step trace 与 fixed-CFL 50-step
  stability。manifest 和可用节点证据未齐前，继续关闭多 zone HIP
  gradient/reconstruction/state-update/boundary/time-step opt-in，也不更新正式
  performance report。
- `b33b7bc0` 已同步到 Kunshan；root HIP build 作业 `124989236` 在节点
  `e04r1n00` 完成（`COMPLETED`, exit `0:0`）。`dcu-preflight.txt` 报告
  `DCU_PREFLIGHT_PASS expected_arch=gfx906`，随后 root HIP smoke 与硬件 CTest
  `16/16` 通过。该结果确认本轮 preflight/build 环境可用，但不替代真实
  8-zone HIP trace 或 stability gate。
- 同一 root binary 在保留的 8-zone case、`e04r1n00`、8 ranks 上完成
  3-step CPU/HIP trace（job `124995642`）：runner 返回
  `MULTIZONE_TRACE_PASS ranks=8 steps=3 cfl=0.01`，8 个 rank 均有
  `STAGE_TRACE_PASS records=36`。但 `fixture-validation.txt` 仍为
  `MULTIZONE_FIXTURE_UNCHECKED metadata=unset`，所以这不是 ownership
  metadata gate。
- fixed-CFL 50-step stability 尝试（job `124996224`）在 legacy、CPU batch、
  HIP batch 三种模式均以 exit `139`/`SIGSEGV` 结束，发生在 8-rank 网格/
  face-topology 初始化阶段；没有把它记为数值或 DCU runtime failure。
  短 trace 通过而 stability 初始化崩溃，下一步应检查 stability runner 对
  partitioned case 的准备和路径契约，再重跑；在此之前 stability gate 仍是
  FAIL/未闭合，不得更新性能报告。
- 对 `root-b33b7bc0/bin/OneFLOW` 的崩溃地址做了 `addr2line`：入口为
  `ONEFLOW::Partition::ReadGrid -> Zone::GetGridReference`，确认问题发生在
  stability case 的 partition workflow 初始化，而非 HIP kernel、MPI halo
  payload 或非物理诊断。后续修复应先让 stability runner 明确接收已经分区的
  8-zone case，并避免再次触发不适用的 repartition path。

## 2026-10-20 follow-up: partitioned stability runner repaired

- `9c2870bc` 修复 `f2-main-solver-stability.slurm` 的 case preparation：
  stability runner 现在在 `ONEFLOW_F2_STABILITY_PARTITIONED_CASE=1` 时显式写入
  `simutask=Solve`、已分区 grid 文件和 boundary map，并把 case 绝对路径传给
  solver；不再把 fixture 原有的 `simutask=Grid` 当作求解入口，避免再次进入
  `Partition::ReadGrid`。同时加入可选的
  `ONEFLOW_F2_STABILITY_FIXTURE_METADATA`/`ONEFLOW_MZ_FIXTURE_METADATA` gate，
  结果写入 `fixture-validation.txt`。
- Kunshan 第一次复核（revision `9c2870bc`，job `125000888`）确认修复方向：
  legacy 与 CPU batch 均为 8/8 active ranks、`exit_code=0`、无诊断命中；
  HIP batch 先因 runner 未开启 `ONEFLOW_ENABLE_UNS_HIP_INTERFACE=1` 被应用
  明确拒绝，属于配置失败而非 HIP 数值失败。
- `1d7ad065` 补齐 partitioned stability 的 HIP interface opt-in，并同步到
  Kunshan。job `125001559` 在节点 `b16r1n00` `COMPLETED/0:0`；fixed-CFL
  `0.01`、50 steps、8 CPU ranks/8 HIP ranks 下，legacy、CPU batch、HIP batch
  三路均为 `exit_code=0`、`diagnostic_hits=0`、`PASS`，每路 8 份
  breakdown 均生成且 rank utilization 通过。
- 该作业的 `fixture-validation.txt` 仍为
  `MULTIZONE_FIXTURE_UNCHECKED metadata=unset`。因此 stability 的
  solver/physicality/rank-utilization gate 已重新闭合，但 ownership/interface
  metadata gate 仍未闭合；不能把这次结果升级为 `MULTIZONE_FIXTURE_PASS`。
- 当前唯一未决的多 zone 证据是由 fixture owner 在 Kunshan 私有 case 上导出
  schema v1 manifest（zone owner、owned/ghost cell、reciprocal endpoint
  face/cell、equation-major counts、memory space），然后重新执行 verifier、
  CPU/HIP trace 和 stability。未取得该 manifest 前不更新正式性能报告。

## 交接摘要（历史，2026-10-18）

> 本节已被文首 **`## 交接摘要（先读这一段，2026-10-23）`** 取代，内容（PR #159/#160/#203 状态、
> 2026-09/10 的单 zone 原始 opt-in 验收、旧的 Kunshan 度量）仅作历史记录，**不要据此判断当前能力**。
> 当前结论：受限 inviscid 路径的七个阶段已默认 device 化、多 zone 已闭合、100-step 端到端约 2.4–2.7x。

## Big picture：正确性优先的统一 GPU solver 架构

目标不是分别维护“1D GPU solver”和“3D GPU solver”，而是在 CPU oracle 保持可信的
前提下，由 `codes/accel` 提供一套 1D/3D 共用的 backend、state、view、lifecycle、
trace/stats 和设备所有权 contract；1D port 是小规模数值与生命周期验证客户端，3D
主 solver 通过 adapter 使用同一 contract。2026-10-10 的 `b980b326` 已将受限
local inviscid timestep 接入同一 state-owned HIP chain；3-step 36-record trace、
fixed-CFL `0.01` 50-step stability 和同资源 25-step paired timing 均已闭环，
但 capability 仍限于单 zone/finest grid/5 方程/Lax-Friedrichs/limiter-off/inviscid。

- [x] 保留 CPU normal/strict 与 stage trace 作为数值 oracle，不以 build-only 或单一
      checksum 代替正确性。
- [x] 在 `codes/accel` 建立 backend-neutral views、`CreateState/Upload/Advance/Download`
      生命周期、state registry 和 opaque backend-specific ownership seam。
- [x] 3D 受限 inviscid 路径已接入
      gradient → reconstruction → flux/residual → state-update 的连续 HIP seam。
- [x] 建立 `ports/kunshan/oneflow_1d_hip` → `codes/accel::EulerDomainBackend` 的薄
      lifecycle adapter；CPU direct/adapter 最终 state 完全一致，专项 contract `3/3`，
      且标准 CPU normal/strict 五算例各 `5/5`。
- [x] 让 1D HIP/stateful benchmark 也通过 shared adapter 运行，并补齐 FullTrace、
      stats、device identity 和性能等价；现有 1D CPU/HIP direct path 继续保留为 oracle，
      暂不切换正式入口。
- [ ] 在现有 reconstruction/connectivity views 上补齐共用 boundary evaluation 与 ghost
      update contract，覆盖 ordinary、solid、interface、periodic、ghost ownership 与
      physicality；先用 1D/小 case 验证，再接 3D，不再创建平行 view 体系。
- [ ] 让 production NoTrace 路径跨 RK stage 保持 `q`、gradient、reconstruction、
      residual 和更新后 state 驻留，消除只为 host boundary/oracle 存在的常规 D2H/H2D。
- [ ] 把 topology/geometry/boundary/halo generation 与 solver/zone/grid/backend/device
      key 统一到唯一 state owner；禁止再增加 process-wide 或 port-local 的第二套
      persistent GPU state。
- [ ] 完成多 zone、MPI halo、单节点多卡/跨节点映射与错误传播；CPU/MPI oracle 和
      workload exit code 必须同时通过。
- [ ] 在同一资源口径下完成 accuracy、stability、MPI correctness 和多次 paired timing；
      只有应用级 wall-clock 重复显示收益，才宣称 GPU 加速并更新正式性能报告。
- [ ] 在 inviscid contract 闭合后，再扩展 viscous/turbulence；不以 1D microbenchmark
      的加速比外推完整 3D Navier–Stokes。

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

1. PR #159 / #160 已于 2026-09-19 合并并通过 CI；fork topic branches 已于 2026-10-09 清理（见 §2 P0）。
2. 保持新 reconstruction seam 为显式 opt-in；继续保留 CPU legacy/batch 和旧 host-staged HIP fallback。
3. 下一刀优先消除 gradient D2H：让 reconstruction view 直接消费 state-owned device gradient，同时保留 MRField 下载作为 CPU oracle/FullTrace fallback。
4. 再迁移 device residual/state update；先确认 `LOAD_RESIDUALS → UPDATE_RESIDUALS → CALC_LHS → UPDATE_FLOWFIELD` 契约，不混入 viscous/turbulence、MPI/halo 或 RK algorithm change。
5. 为 interface/periodic/solid/ordinary boundary、ghost gradient copy、qf1/qf2 trace 补充小 case contract；记录 H2D/D2H bytes、kernel launches、sync 和 allocation。
6. 继续按 CPU oracle/trace → 3-step accuracy → fixed-CFL 50-step stability → 多次同 basis timing 验收；只有 accuracy/stability/MPI correctness 和重复 timing 全通过后才更新正式报告。

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
- **坑**：upstream 旧版 `euler-dcu-gtest.slurm` 在 `module purge` 之前解析 cmake 且不传 `-Damd_comgr_DIR`，在集群上必失败；本轮用的是 dev 上已修好的副本，PR #160 已把同一修复带入。CPU/MPI 与 root main-solver 测试固定使用 Kunshan 默认依赖；HIP 专用流程的 CMake/host compiler 兼容设置见对应脚本和记录，不要把旧模块配置带入 CPU/MPI 测试。

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


## 本轮 HIP reconstruction residency seam（2026-09-22）

- 新增 `CellFaceReconstructionView` 的 HIP consumer seam；`ReconstructFaceEulerKernel` 保持 CPU `GetQlQr`、face-center 到 left/right cell-center 位移、limiter-off `phi=1`、physicality fallback、boundary-first、`INTERFACE/PERIODIC` preserve、普通 boundary average、`SOLID_SURFACE` 的 face-indexed `bc_q` 覆盖语义。
- `HipGradientStorage` 现在 state-own `qf1/qf2`、boundary operation/mask 和可选 `bc_q`；HIP fused flux/residual 直接读取 device qf1/qf2。`ONEFLOW_ENABLE_UNS_HIP_RECONSTRUCTION=1` 是新的受限 opt-in，旧 host reconstruction 和旧 HIP batch fallback 保留。
- NoTrace 不做 qf1/qf2 或 invflux D2H；只有 trace/stage trace 显式下载 qf1/qf2/invflux。当前 residual 仍 H2D/D2H，RK/state update 仍 host loop；q H2D、gradient D2H 仍保留用于 oracle/diagnostic，故不能称完整 device-resident solver。
- Kunshan revision 使用 DTK 26.04 / `gfx906` / `dcu:1`：root configure/build、smoke、GoogleTest `9/9`、hardware CTest `10/10` 全通过。
- 同一 revision 的标准 CPU 回归也通过：`kshcnormal`、16 CPU、normal `1e-8` 五算例 `5/5`，strict `1e-15` 五算例 `5/5`，Slurm `COMPLETED/0:0`。
- 直接单进程 m6 1-step trace（legacy / CPU batch / HIP batch + gradient + reconstruction）三路 workload exit `0`；stream verifier `STAGE_TRACE_PASS records=12`，HIP 相对 legacy 最大绝对差约 `2.3981e-13`，finite/positive density-pressure 和 connectivity/geometry metadata 全通过。
- fixed-CFL `0.01`、50-step 直接 workload 三路均 exit `0`，精确诊断 regex 三路 `diagnostic_hits=0`。该包装 job 因初版宽松 grep 把日志中的 “information” 误计为命中而返回 Slurm `FAILED/1:0`，所以不把它记为标准 stability runner PASS；workload 本身没有失败，也未做 timing。
- 当前能力仍受限于单 zone、finest grid、5 方程、Lax-Friedrichs、limiter off、inviscid；viscous/turbulence、MPI/interface、RK/state update、完整 q residency 和 GPU reduction 尚未迁移。

## 本轮 HIP gradient ownership migration（2026-09-22）

- `HipFluxBackend` 新增 HIP-specific `Ns3DBackendState`，production `CalcGradient` 按 grid binding 选择 state-owned `cellState`、gradient、geometry 和 connectivity DeviceBuffer；未绑定 direct/smoke API 保留 shared fallback。
- bind/unbind 在 `EulerDomainStateSync` 的 HIP state create/invalidate/restart 路径接入；backend/device identity 校验保留，cache 失效同时比较 state owner token、topology generation 和 view extent/pointer。
- Kunshan CPU 计算节点：5 个 Euler domain/registry contract binaries 全部通过（`4+9+2+3+8` tests）；DTK 26.04 / `gfx906` / `dcu:1`：root HIP configure/build 通过；smoke PASS；GoogleTest `9/9`；hardware CTest `10/10`。本轮未改变 kernel 数值语义、H2D/D2H 边界、allocation 或 sync 统计，未做性能 timing。
- 本轮 root HIP 作业的三种 3-step workload 均生成完整 trace/log，但标准 `mpirun -np 1` runner 在 PMIX cleanup 阶段未返回，作业被取消，故该 revision 不把 stage verifier 或 workload exit 记为通过；此前同一基线 revision 的 3-step/50-step CPU/HIP 门禁证据仍有效。
- 当前能力仍受限于单 zone、finest grid、5 方程、Lax-Friedrichs、limiter off、inviscid；reconstruction、qf1/qf2、flux/residual、RK/state update、viscous/turbulence 与 MPI/interface 尚未 device-resident。

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

## 2026-09-23 host boundary 优化（FarField fast path）

- `NsCalcBc` 的低扰动拆分和昆山 `perf` 采样已完成：m6 的 `boundary_calc_face_bc` 约占 boundary 时间的绝大部分；采样中约 76% 落在 `NsBcSolver::FarFieldBc` 的标量 `pow`，而 `SetId`、`PrepareData`、`UpdateBc` 合计仅约几十毫秒。
- 新增了同一 boundary sweep 内的 FarField 参考声速/参考熵缓存，并对 `vismodel==INVISCID` 且 `gamma≈1.4` 的正数物理量，将 `pow(x, 1/(gamma-1))` 改写为 `x*x*sqrt(x)`；其他 gamma 和全部 viscous 路径继续使用原始 `pow`，保留 CPU oracle 语义。
- 昆山 m6、`steps=3,warmup=1,repeats=1`、同资源口径的探索性结果：HIP 端到端约 `13.16 s → 2.76 s`；`boundary_bc` 约 `10.18 s → 0.52 s`；`boundary_calc_face_bc` 约 `9.64 s → 15 ms`。该结果为单次探索性作业，不能直接替代正式性能报告的多重复统计。
- 随后补做同口径 `warmup=1,repeats=3`：legacy CPU 8-rank mean `5885.950 ms`，HIP 1-rank mean `2772.724 ms`，端到端 ratio `2.1228x`；HIP `boundary_bc` 三次为 `534.871/532.728/535.757 ms`，`boundary_calc_face_bc` 为 `17.551/15.734/16.276 ms`。作业 workload 三路均 exit `0`，Slurm `COMPLETED/0:0`；仍属于 inviscid-only、3-step 探索性口径，不更新正式性能报告。
- 随后发现上述 `boundary_bc≈0.53 s` 仍包含逐面 profiler 的 `getenv + mutex + strcmp` 扰动；已改为每个 region 本地累计、循环结束只记录一次，并在默认无 profiler 时走无计时分支。修正后同口径 `warmup=1,repeats=3`：legacy CPU 8-rank mean `5500.673 ms`，HIP 1-rank mean `2424.967 ms`，ratio `2.2683x`；HIP `boundary_bc=101.993/102.329/106.150 ms`，`boundary_calc_face_bc=15.364/15.413/15.380 ms`。因此 host boundary 已不再是端到端主瓶颈，下一刀转向剩余 host-staged 数据链。
- 昆山 CPU normal/strict 五算例在带 DTK 运行时的验证作业中均为 `5/5`；strict 最大绝对残差 `1.1072414686508214e-17`。第一次 CPU 队列失败是 HIP 构建缺少 `libamd_comgr` 的环境问题，不计入数值结论。
- fast path 后的标准 3-step HIP trace 已生成 legacy/CPU batch/HIP batch 三套完整 stage 文件；流式 verifier 在处理约 5 GB trace、sequence 6/12 时为释放计算资源主动取消，因此本轮不把它记为完整 verifier PASS。各 workload 日志已显示 finite/positive state，boundary 无 trace workload 与 CPU normal/strict 门禁均通过。25-step wrapper 因大型 trace runner 未形成完整 `result.txt`，不作为正式 25-step 证据。
- 后续优化方向：若要覆盖 viscous/FarField，需设计满足 strict oracle 的近似或查表方案；不要直接把 `x*x*sqrt(x)` 扩展到 viscous 路径。对不同 gamma、边界类型和动态热化学模型继续保持原始公式。

## 2026-09-24 boundary field-pointer cache

- `UNsBcSolver::Init()` 现在为当前 zone/grid 缓存 equation-major `q`、`bc_q`、`tempr`
  和 `gama` 的连续指针，同时缓存边界 face 的 connectivity、几何量和 cell-center
  指针；`SetId()`、`PrepareData()`、`UpdateBc()` 复用这些指针，未改变边界类型分支、
  boundary-first 顺序、ghost ownership 或 CPU 数值公式。
- Kunshan CPU oracle 已通过：normal `1e-8` 五算例 `5/5`，最大绝对残差差
  `4.970574442764598e-10`；strict `1e-15` 五算例 `5/5`，最大绝对残差差
  `1.1072414686508214e-17`。
- Kunshan DTK 26.04 / `gfx906` / `dcu:1` 已通过 root HIP configure/build、smoke、
  GoogleTest `9/9`、hardware CTest `10/10`。完整
  gradient → reconstruction → residual → state-update 3-step seam 为 36 records PASS；
  HIP 对 legacy 最大绝对差 `3.7747582837255322e-13`，metadata 一致，state
  finite 且 density/pressure 为正。
- 随后原型实现了 function-local static boundary solver 与 region face/type/name/data
  metadata cache。该原型也通过同样的 CPU/HIP correctness gate，但同一 Slurm
  allocation 的 paired timing 没有显示稳定收益，故已撤销：
  - basis：`steps=3`、`warmup=1`、`repeats=3`、legacy CPU 8 ranks、HIP 1 rank、
    `dcu:1`；
  - pointer-cache A：legacy mean `5251.237 ms`，HIP mean `2008.560 ms`；
  - region-cache B：legacy median `5320.089 ms`，其中一次 `9302.341 ms` 异常值；
    HIP mean `2002.470 ms`；
  - HIP 的 B/A 仅 `1.003041x`，三次 paired 方向不一致，不构成可重复收益。
- 当前工作树保留 pointer cache，三个相关源码文件与已验证 A 版本的 SHA-256 完全
  一致。下一步转向 1D/3D 共用 boundary/ghost view 与 device ownership contract，
  不继续为 region metadata 增加 host-only 缓存层。

## 2026-09-24 1D/shared EulerDomain lifecycle adapter

- 新增 `OneDEulerDomainAdapter`，把现有 1D `EulerBackend/EulerState` 包装为
  `codes/accel::EulerDomainBackend/EulerDomainState`；adapter 不持有第二份数值或设备
  state，只负责 problem、boundary、state key、run options 和 field view 的契约转换。
- 当前受限 contract 明确要求 3 方程、无外部 ghost、Periodic/Outflow（映射为 1D
  Transmissive）、adapter 自己管理内部 RK stages；不把 3D stage callback、5 方程或
  Wall/Inflow 静默映射到 1D。
- 新增 CPU contract tests：
  - shared adapter 与直接 `CpuEulerBackend` 连续 2 steps 的最终 state 逐元素完全一致；
  - 拒绝不支持的 equation/ghost/boundary/backend-device ownership；
  - 拒绝外部 RK stage scheduling，避免共享 contract 与 1D 内部 RK 重复推进。
- Kunshan `kshcnormal` 验证通过：完整 root build 成功，CPU normal `1e-8` 五算例
  `5/5`，strict `1e-15` 五算例 `5/5`；adapter GoogleTest `3/3`，两个作业的
  scheduler/workload 均为 `COMPLETED/0:0`。
- adapter state ownership 现在同时校验 wrapped backend、backend kind 和 device id；
  不允许同一个 `HipEulerBackend` 的不同 device-id adapter 串用 state，也不允许把 HIP
  backend 声明成 CUDA/Kokkos。
- Kunshan DTK 26.04 / `gfx906` / `dcu:1` 验证通过：
  - standalone HIP GoogleTest/CTest 均为 `15/15`；
  - root HIP build 与 smoke 通过，GoogleTest `15/15`、hardware CTest `16/16`；
  - HIP adapter NoTrace 最终 state、FullTrace 的 face state/flux/residual/RK state、
    stats、连续两次 lifecycle advance 与 direct `HipEulerBackend` 一致；
  - 三个作业的 scheduler/workload 均为 `COMPLETED/0:0`。
- 新增 `oneflow_1d_euler_domain_adapter_benchmark`，在同一 allocation 内交错运行 direct
  HIP 与 shared adapter。basis 为 `nx=1048576`、`steps=100`、`repeats=5`、
  `warmup=2`：direct/shared `Advance` 总时间分别为 `420.163270/420.126780 ms`，
  shared/direct=`0.999913`；kernel 时间分别为 `417.611221/417.556526 ms`，
  launches `3000/3000`、syncs `5/5`，最终最大绝对误差 `0` 且 checksum 完全一致。
  这是 adapter 薄层等价性证据，不外推为 3D 应用级加速。
- 下一步转向 1D/3D 共用 boundary evaluation/ghost update contract；保留 direct
  CPU/HIP oracle，不切换 1D 正式入口，也不新建第二套 boundary view 或 GPU state。

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
| 主分支 | `master` = `origin/master` = `upstream/master` = `b83d78c7`（2026-10-23 三端同步；已合入 dev） |
| upstream PR | #159/#160 已 MERGED；#203 仍只等待 review/CI，不作为当前开发阻塞项 |
| 分支 | `dev` = `origin/dev` = `762a847e`（merge `master` into `dev`，无冲突）；领先 `upstream/master` 264 个提交 |
| 昆山工作区 | 已规范化：`<workspace>/` 下 `src/`、`deps/`、`builds/`、`runs/<date>/<suite>/`、`archive/`；集群侧 README 记录具体路径 |
| 昆山作业脚本 | 四个标准套件脚本已更新到新工作区路径 |
| 智能体入口 | 仓库 `AGENTS.md`（含文档地图、分支模型与工作规则）；`CLAUDE.md` 已于 2026-09-19 删除；技能仓库 `oneflow-dev`（已安装到本地 skills 目录） |
| 测量口径 | 已确立：`lifecycle_*_ms` 为 repeats 总和，异口径不可比；历史 13.10× 勘误已修正为 25.55× |
| 当前进度 | 受限 inviscid 路径（5 方程 / Lax–Friedrichs / limiter-off / 显式 RK）的七个阶段已**默认** device 化并通过门禁：fused flux+residual、interface/halo、gradient、reconstruction、state update+resident residual、device boundary、resident time-step。多 zone（8 zone/8 rank）与单 zone 均已闭合；NoTrace 残差/正性检查由设备归约标量提供。未迁移：viscous/turbulence、limiter、multigrid、LUSGS、双时间。 |
| 最新验证 | `52fe404d`（2026-10-23，Kunshan DTK 26.04 / gfx906）：8-zone 3-step trace `MULTIZONE_TRACE_PASS`（每 rank 36 records，最大绝对差 `2.284e-13`）、resident contract `24/0/0`、CPU/HIP 100-step 运行 `exit_code=0` 且 rank utilization PASS、restart `50+50` vs `100` 通过（CPU 逐位一致，HIP 流场最大相对差约 `1.6e-14`）。同 basis 8-zone：8C CPU `21.60 s` vs 8C+1DCU `9.08 s` = `2.38x`；同节点隔离 `28.40 s` vs `9.08 s` = `3.13x`。详细口径见 §5 与新增的 2026-10-23 收口节。 |

**能力边界（不要越界声明）**：3D 主 solver 的 HIP 能力目前严格限定在
`NS_SOLVER` / 5 方程 / Lax–Friedrichs / `ilim=0` / `vismodel=0` / 单网格层 / 显式多阶段 RK /
1 rank 1 zone / `nGhostCells==nBFaces`；这七个阶段默认走 device，其它配置自动回退 host。
halo 传输仍是 HostStaged（DeviceDirect 实测更慢，保持关闭）。多卡 8:1 核卡比受限于缺少
≥64-zone fixture（`Partition` 任务未实现）。跨节点已有正确性证据但无规范核卡比。
viscous/turbulence/SA、limiter、multigrid、LUSGS、双时间未迁移；不能称为完整
stateful/device-resident NS solver。

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

- [x] **PR #159**（`pr/agents-branch-model`）：已 review approved 并于 2026-09-19 合并；CI 4/4 通过。
- [x] **PR #160**（`pr/euler-weno5-unified`）：已 review approved 并于 2026-09-19 合并；PR 分支自身 T1 `cpu-regression` + T2 `dcu-single` 门禁通过，CI 4/4 通过。
- [x] 两个 PR 合并后清理对应 topic 分支：本地无分支，2026-10-09 已删除 origin 上的两个分支。
- [ ] **PR #203**（`pr/database-type-map-init`）：修复 `DataBaseType` 映射在首次访问前未初始化的问题；Kunshan 冷启动回归 1/1 通过。等待 Linux/Windows CI 与 review；通过后再按 upstream 结果收口并清理 topic branch。

### P0 — dev 融合基线

- [x] 最新 `origin/master` / `upstream/master` 已同步到本地 `master`。
- [x] 协同开发 Phase 1-3、旧架构 contract/StateRegistry 已统一恢复到本地 `dev`。
- [x] standalone CPU、根工程、根 CTest、CPU bridge 已完成本机验证。
- [x] 完成融合后 dev 的 contract/adapter 验证并更新 `origin/dev`。
- [x] 已将 upstream master 合并到 dev；保留 mutable `SimuContext&` 与 StateRegistry 生命周期，并吸收 upstream 的分阶段 `FieldSimu`、policy 和 parser 测试。2026-10-09 核对时本地 `master` 与 `origin/master`、`upstream/master` 一致，本地 `dev` 与 `origin/dev` 一致；具体 commit 以当前 refs 为准。

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
    - [x] F3.5：2026-09-22 对包含 P0 contract 与 HIP gradient ownership wiring 的新 revision 完成 Kunshan CPU normal/strict 五算例各 `5/5`（workload/scheduler `0:0`/`COMPLETED`），以及 root HIP configure/build、smoke、GoogleTest `9/9`、hardware CTest `10/10`；3-step 三路 workload 生成完整 trace/log，但标准 `mpirun -np 1` runner 在 PMIX cleanup 阶段未返回并被取消，因此该 revision 不记 stage verifier/workload exit 为通过，也不产生性能结论。
- [x] 昆山回归 eric 的完整 `task/database/register/adt` 测试套件：GCC `16.2.0` / CMake `4.4.3` / OpenMPI `5.0.11`，145 tests 全部通过。

### P2 — 后续技术工作

说明：先完成可观测性和 host-staged 路径优化，再做设备归约、WENO5 DCU 验证和性能优化。每项性能改动都必须在同一输入、同一 `steps/warmup/repeats` basis 下比较，并保留 accuracy gate 结果。

- [x] **P2.0：主 solver stage 分段计时**
  - [x] `StageProfiler` 已在初始化、gradient、limiter、reconstruction、boundary reconstruction、host pack、geometry/connectivity H2D、H2D、HIP kernel、D2H、viscous/turbulence、RK update、residual/state update、MPI/interface 和 output 边界提供 opt-in 计时。
  - [x] `ci/kunshan/f2-main-solver-stage.slurm` 输出 machine-readable `breakdown.tsv`；正式汇总只读取 `repeat-*`，与 `steps=3,warmup=1,repeats=3` 的 timing basis 分开且不混入 warmup。
  - [x] 2026-09-20 Kunshan target-node 已完成 inviscid-only breakdown：51 个正式 repeat/rank 文件、0 条 warmup 行、所有数值 finite。HIP 的 per-repeat mean 为 `gradient=1046.610 ms`、`reconstruction=921.434 ms`，显著高于 `H2D+D2H+HIP kernel+geometry H2D` 约 `129.805 ms`；下一迁移路线据此选择 gradient/reconstruction。`rk_update` 包含嵌套阶段，不能与子类别相加；viscous/turbulence 本轮禁用，完整 NS 仍需另测。
  - [x] 2026-09-23 新增 `LOAD_Q/CALC_TIME_STEP/LOAD_RESIDUALS/UPDATE_RESIDUALS/CALC_LHS/UPDATE_FLOWFIELD/CALC_BOUNDARY` 子阶段计时，并在昆山同一 25-step 作业上完成 `warmup=1,repeats=2` 的 legacy CPU 8-rank/HIP 1-rank profiling。正式 wall-clock 均值为 legacy `103789.071 ms`、HIP `81288.434 ms`（探索性 ratio `1.2767x`，不更新正式性能报告）。HIP 的 `CALC_BOUNDARY=72392.550 ms`，占 HIP wall-clock `89.1%`；`UPDATE_RESIDUALS=3907.425 ms`（4.8%），gradient+reconstruction 约 `1828.803 ms`（2.2%）。legacy 的 `CALC_BOUNDARY=70867.800 ms`、`UPDATE_RESIDUALS=26281.050 ms`。结论：继续增加迭代步数不会显著放大收益，下一刀应先拆分并迁移 `NsCalcBoundary`（gamma/viscosity/`NsCalcBc` 及 ghost 回写）的 host path；不要先继续微调已迁移的 flux/gradient/state kernel。

- [ ] **P2.1：把 HIP backend/state 绑定到宏步生命周期**
  - [x] `HipFluxBackend::Shared()` 已让同一 runtime 内的 flux/gradient 调用复用长期 backend 与 device buffers，不再在每次 `CalcInvFlux` 创建临时 backend。
  - [x] `SimuContext::TeardownEnvironment()` 在 accelerator runtime finalize 前先 clear registry state、再 `ReleaseShared()`，旧 teardown signal 11 已消除。
  - [x] 建立 backend-neutral reconstruction view、opaque state-owned backend seam、generation/token，以及包含 backend/device identity 的 state key；补齐 create/reuse/invalidate/restart/teardown contract tests。
  - [x] 绑定 `Ns3DDeviceState` 的 host-staged primitive flux 入口现在把面状态、几何/连通性、boundary mask、residual 和可选 face flux 放入该 state 的 HIP storage；未绑定 direct/smoke API 继续使用 backend fallback，避免改变 CPU/standalone oracle。
  - [ ] 该 ownership 收敛尚未在本机编译 HIP；需在 Kunshan DTK 目标节点完成编译、3-step accuracy、fixed-CFL stability、teardown/reinitialize 和 MPI correctness 复验后再关闭本项。
  - [ ] 把实际 shared backend/device buffers 收敛到按 solver/zone/grid/backend/device key 管理的 `Ns3DDeviceState`，补 teardown/reinitialize contract，形成唯一且可失效的 device ownership。
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
  - [x] 验收：主 solver HIP batch 的 residual 回写确实走 device kernel，并通过 conservation、state trace 和 50-step stability gate。

- [x] **P2.5：HIP Green–Gauss gradient vertical slice**
  - [x] 新增 equation-major cell/ghost gradient contract、显式 opt-in 与 capability fail-fast；保留 CPU Green–Gauss oracle。
  - [x] geometry/connectivity cache、device accumulation/normalize/boundary-copy kernel 与 synthetic CPU-oracle smoke 已完成。
  - [x] 本地 build/CTest、Kunshan target-node contract、3-step accuracy 和 fixed-CFL 50-step stability 全通过。
  - [x] 同 basis 单作业显示 HIP `gradient` 由约 `1046.610 ms` 降为 `176.883 ms`，端到端 HIP/legacy 为 `1.160129x`。
  - 边界：当前仍每次 q H2D、gradient D2H，且 device buffers 尚未由 `Ns3DDeviceState` 唯一持有。

- [x] **P2.6：HIP reconstruction vertical slice（受限主路径已闭环；扩展 contract 继续跟踪）**
  - [x] **P2.6.1：** 完成 limiter-off reconstruction 的 MRField/device contract、face ownership、boundary reconstruction 顺序与 trace oracle 只读盘点；不先写仍需 gradient D2H + qf H2D 的孤立 kernel。
  - [x] **P2.6.2：** 建立 backend-neutral cell/face reconstruction view，并把首批 q/gradient/qf1/qf2/geometry/connectivity/residual 的 ownership、generation 与失效边界纳入按 solver/zone/grid/backend/device key 管理的 state-owned contract；CPU/HIP 类型仍通过 opaque seam 隔离。
  - [x] **P2.6.3：** production gradient 的 HIP state/geometry/connectivity DeviceBuffer 已迁入上述 state-owned backend-specific seam；qf1/qf2、flux/residual/state update 已接入连续 device chain，并在目标节点完成编译、accuracy 与 stability 验收。
  - [x] **P2.6.4：** 当前垂直切片严格限制为单 zone、finest grid、5 方程、Lax-Friedrichs、limiter off、inviscid；未同时迁移 limiter、RK algorithm 或 viscous/turbulence。
  - [x] **P2.6.5：** 已按 3-step accuracy → fixed-CFL 50-step stability → 同 basis timing 顺序完成受限路径验收；2026-09-23 的探索性 timing 不作为稳定性能结论，也未更新正式报告。
  - [x] **P2.6.6：** 在最终 checkpoint 上修正/复核 stability wrapper，使用精确 diagnostic regex，已闭环 legacy、CPU batch、HIP reconstruction/state-update 三路 50-step 的 workload exit、diagnostic hits 和 scheduler state。
  - [ ] **P2.6.7：** 补齐 interface、periodic、solid-surface `bc_q`、ordinary boundary、ghost gradient copy 与 generation/token cache invalidation 小 case contract。
  - [ ] **P2.6.8：** 继续比较 qf1、qf2、invflux、residual、state 和 connectivity/geometry metadata；NoTrace 不下载完整诊断数组。

- [ ] **P2.7：重新测量端到端性能**
  - [x] **P2.7.1：** 已固定 `steps=3,warmup=1,repeats=3`、m6 输入、CPU 8 ranks、HIP 1 rank 和同一资源 tuple，完成 gradient slice 的首个正式单作业测量。
  - [x] **P2.7.2：** 首个阶段目标 `HIP batch < legacy CPU` 已在该单作业达到，但不提前承诺稳定倍数。
  - [ ] **P2.7.3：** 只有连续多次测量方向一致且 accuracy/50-step/MPI correctness 全通过，才更新性能报告。
  - [ ] **P2.7.4：** 让 device gradient view 成为 HIP reconstruction 的生产输入，MRField gradient 下载只保留给 CPU oracle、FullTrace 和 diagnostic fallback，并记录 allocation、H2D/D2H、kernel launch、synchronize 与 stage wall-clock。
  - [ ] **P2.7.5：** 不用 async API 掩盖未定义依赖，先保持明确 stage fence。
  - [x] **P2.7.6：** 2026-09-23 细粒度 profiling 已证实当前端到端 HIP 瓶颈不是短运行摊薄，而是每个 RK stage 的 host `CALC_BOUNDARY`；该类别在 25-step HIP wall-clock 中占约 `89.1%`。3-step 子项 profile 显示正式 repeat 的 `boundary_bc≈9422 ms`，而 `boundary_gamma_inner≈73 ms`、`boundary_gamma_ghost≈1 ms`、inviscid viscosity 为 `0`；因此下一刀应直接评估 `NsCalcBc` 的 device boundary/ghost update，继续微调 gamma、flux 或 state kernel 不会改变端到端瓶颈。guard `vismodel=0` 下跳过 `CalcLaminarViscosity` 的复测未带来可分辨收益（25-step HIP mean `81.288 s→81.247 s`，在跨作业波动内），保留为语义清理而非性能结论。
  - [x] **P2.7.7：** 2026-09-23 完成第一轮 host lifecycle 优化：`UNsInvFlux` 跨 RK stage 复用 `qf1/qf2/invflux`，并在 HIP reconstruction 中稳定保存 boundary operation host buffer；backend 仅在 operation host pointer/extent 变化时重传静态元数据，`bc_q` 数值仍按 stage 更新。同步将 `NsCalcGamaT` 的内部/ghost field 访问改为连续 `data()` 指针，保持 CPU oracle 计算顺序不变。
  - [x] **P2.7.8：** 目标节点同 basis 探索性复测（m6、`steps=3`、`warmup=1`、`repeats=3`、legacy 8 ranks、HIP 1 rank、`dcu:1`）均 workload `exit_code=0`、Slurm `COMPLETED/0:0`。host-reuse 基线 HIP 正式均值 `2286.562 ms`；加入 boundary operation cache 后为 `1968.529 ms`（跨作业约 `13.9%`，需 paired repeat 才能归因）；随后 gamma pointer 版本为 `1887.007 ms`，但 `boundary_gamma_inner` 仍约 `73 ms`，未观察到可分辨的子阶段下降。因此不更新正式性能报告，下一刀仍是减少 `CALC_BOUNDARY` 的 host face loop 或迁移 boundary/ghost update。
  - [x] **P2.7.9：** 2026-10-10 在 `b980b326` 完成 resident 3D local timestep vertical slice。Kunshan accuracy gate（m6、3 steps、fixed CFL `0.01`、36 条 trace、gradient/reconstruction/state-update/boundary/timestep 全开）三路 workload 与 scheduler 均通过；`STAGE_TRACE_PASS records=36`，legacy→HIP 最大绝对差 `2.3958612871410878e-13`，inviscid residual 最大差约 `3.6e-15`，face/cell metadata 差异为 `0`，状态始终 finite 且 density/pressure 为正。fixed-CFL `0.01`、50-step stability 三路均 `exit_code=0`、`diagnostic_hits=0`、`PASS`，scheduler `COMPLETED/0:0`。
  - [x] **P2.7.10：** 2026-10-10 使用同一 `kshdnormal` 单卡资源和 `steps=25,warmup=1,repeats=3` 完成 baseline/candidate HIP 交替配对 timing；正式作业 `COMPLETED/0:0`、12 次 workload exit 均 `0`。`c30695a7` HIP mean `5961.450325 ms`，`b980b326` HIP mean `4794.353067 ms`，candidate/baseline `1.243432x`；同作业 CPU legacy mean `32389.489920 ms`，legacy/candidate `6.755758x`。独立首轮 runner 因未加载 Python 在汇总阶段返回 `127`，其 12 次 workload 仍全为 `0`，仅作诊断，不替代成功作业。
  - [x] **P2.7.11：** stage breakdown 归因确认收益主要来自 resident timestep：`calc_time_step` 三次均值约 `991.014 ms → 26.585 ms`（约 `37.3x`）；`update_residuals` 约 `1090.080 ms → 1057.947 ms`，未发生计时挪移。候选相对旧 HIP 的端到端收益约 `24.3%`，但该证据仍限于单 zone/finest-grid/5 方程/Lax-Friedrichs/limiter-off/inviscid，尚不更新正式公开性能报告。
  - [ ] **P2.7.12：** 在不改变 `repeats`、资源和 case 的前提下再做至少两轮成功 paired timing，并补齐 candidate CPU-MPI/DCU-MPI correctness；只有跨轮方向一致且 MPI/多 zone 门禁通过，才更新正式性能报告。

- [ ] **P2.8：residual/state update**
  - [x] **P2.8.1：** 只读确认 `LOAD_RESIDUALS → UPDATE_RESIDUALS → CALC_LHS → UPDATE_FLOWFIELD` 的真实数据契约；确认 residual 符号、`timestep/volume/rkcoef` 缩放、primitive↔conserved 更新和 boundary/ghost 回写顺序。
  - [x] **P2.8.2：** 已在当前受限 capability 下完成 state-owned residual、device residual scaling/LHS 和 device q update；Kunshan DTK 编译、3-step accuracy、fixed-CFL 50-step stability，以及 legacy/CPU batch 8-rank 与 HIP batch 1-rank workload correctness 均通过。
  - [x] **P2.8.3：** `UNsUpdate::UpdateFlowField` 的 host cell loop 已增加 HIP kernel 化受限分流；保留 host path 和 capability fail-fast，并已在目标节点完成闭环验收。
  - [ ] **P2.8.4：** 不同时迁移 viscous/turbulence、MPI halo 或改变 RK algorithm。

- [ ] **P2.9：主链稳定后的优化**
  - [ ] **P2.9.1：** 评估 stage-level stream/event 优化。
  - [ ] **P2.9.2：** 比较 residual scatter 的 atomic、CSR 和 segmented reduction。
  - [ ] **P2.9.3：** 增加 device-side finite/positivity/error/conservation reduction。
  - [x] **P2.9.4：** 2026-10-19 在 Kunshan `kshdnormal` 完成双节点、4 ranks、4 DCUs
        GPU-aware MPI capability 与 standalone Euler halo correctness。每节点 2 ranks/2
        DCUs，`gfx906`，`gpu_aware_mpi=1`，`decomposition=PASS`，CPU/HIP hash 完全一致，
        workload exit code 为 0。相同 `steps=100,warmup=1,repeats=2` basis 下，HIP/CPU
        lifecycle speedup 为 `9.35x/35.70x/89.69x/138.08x`（`nx=65536/262144/1048576/4194304`），
        GPU-aware 相对 host-staged halo 为 `1.39x/1.31x/1.19x/1.13x`。这是 standalone
        1D Euler kernel/MPI 证据，不外推为 3D 主 solver 跨节点性能。
  - [x] **P2.9.5：** 2026-10-19 补齐同节点资源对照：单节点 `32 CPU ranks × 1
        core` 对 `1 DCU × 1 rank` 和 `4 DCUs × 4 ranks`，四规模 hash/decomposition/halo
        全通过；HIP/CPU lifecycle 分别为 `1.19x/8.64x/5.60x/7.70x` 和
        `1.21x/12.11x/12.79x/26.53x`。另在双节点完成 `64 CPU ranks` 对 `8 DCUs`
        （每节点 32 CPU ranks、4 DCUs），四规模为 `0.78x/2.12x/6.49x/22.01x`。
        所有比较均为 `steps=100,warmup=1,repeats=2`，CPU 是实际活跃 MPI ranks，
        不是仅申请但未使用的 `cpus-per-task`；这些结果仍限于 standalone 1D Euler。
  - [x] **P2.9.6：** 已按代码契约重设计 rank/DCU 实验。standalone 1D Euler
        没有 zone；`Partition(globalNx, rank, ranks, ...)` 会随 rank 数重新划分同一
        全局 cell array，halo 邻居也按新的 rank ring 重建。因此 CPU `8/16/32 ranks`
        与 HIP `1/2/4 ranks = 1/2/4 DCUs` 必须保持相同 `globalNx`、初始条件、
        `steps=100,warmup=1,repeats=2`，不复用另一 rank 数的 local timing。矩阵 runner
        已扩展为四个网格规模、三种 CPU rank 和三种 DCU rank，并硬检查
        `local_ranks`、`decomposition_active_ranks`、非空 local partition、
        `halo_exchanges_sum == expected_halo_exchanges`、HIP `visible_devices` 与
        DCU 数一致，以及 CPU/HIP final hash 一致。
  - [x] **P2.9.7：** 已确认 3D root solver 不能按 standalone 方式“给单 zone 加 rank”。
        `m6wingroe_sa` 是 single-zone，当前 HIP capability 要求 `ZoneState::nLocal == 1`；
        多 rank 3D 实验必须使用已分区 multi-zone fixture，并按请求的 rank 数重新生成
        `ownerRanks` 与 interface exchange plan，验证每个 rank 有非空 local zone 且
        写出 solver work。未完成该 fixture 前，不报告 3D 的 8/16/32-rank CPU 或
        1/2/4-DCU scaling。


- [ ] **P2.10：GPU reduction**（优化计划阶段 D 唯一剩余项）
  - **P2.10.1：** checksum、最大误差、有限性/正状态检查放到设备端归约，只回传标量。
  - **P2.10.2：** 参考 `doc/plans/oneflow-euler-optimization-plan.md` 阶段 C/D。
  - **P2.10.3：** 验收 Kunshan 四规模 correctness 不变、D2H 占比进一步下降并完成性能复测。

- [ ] **P2.11：WENO5 数值内核的 DCU 四规模验证**（单卡 contract 已完成）
  - [x] **P2.11.1：** `nx=32` contract：WENO5 HIP 与 CPU oracle 对齐。
  - [ ] **P2.11.2：** `nx=65536/262144/1048576/4194304` correctness 与性能复测。

### P3 — 维护与清理

说明：清理临时分支和工作区，并把验证后的协作流程沉淀回技能与文档。


- [ ] **P3.1：** PR #203 合并/关闭后清理其 topic branch（本地 + origin）；PR #159/#160 对应 topic branch 已于 2026-10-09 清理，`fix/contract-test-cmake-path` 已于 2026-09-19 删除。
- [ ] **P3.2：** 昆山 `<workspace>/work/`、`tmp/` 定期清理（均可重建）。
- [ ] **P3.3：** `oneflow-dev` 技能更新流程：改技能仓库 → `git push` → 各环境 `git pull`。

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
| 2026-10-09 | 修复数据库类型映射冷启动初始化，并向 upstream 提交 PR #203；补充独立进程回归测试 | Kunshan OpenAPI：GCC `16.2.0` / OpenMPI `5.0.11` / CMake `4.4.3`，`DataBaseTypeInitTest.AccessorsInitializeMappingsOnFirstUse` 1/1 通过；PR `MERGEABLE`、等待 review，Linux/Windows CI 最近检查时运行中 |
| 2026-10-09 | 在 Kunshan 使用 GCC `16.2.0` / CMake `4.4.3` / OpenMPI `5.0.11` 完成 `tests/adt`、`tests/database`、`tests/task`、`tests/register` 全套回归 | CTest `adt 9/9`、`database 49/49`、`task 58/58`、`register 29/29`，合计 `145/145`；Slurm `COMPLETED/0:0` |
| 2026-10-09 | 完成 P0 upstream PR 跟进：PR #159/#160 均已 review approved、合并且 CI 4/4 通过；核验并删除 fork `origin` 上的两个已合并 topic branch，本地对应分支已不存在 | [PR #159](https://github.com/eric2003/OneFLOW/pull/159)、[PR #160](https://github.com/eric2003/OneFLOW/pull/160)；`git ls-remote --heads origin` 确认两个分支引用均已清除 |
| 2026-10-10 | `b980b326` resident 3D local timestep target-node accuracy/stability/performance 门禁；新增 paired runner 并修复其 Python 模块声明 | Kunshan OpenAPI：GCC `16.2.0` / OpenMPI `5.0.11` / CMake `4.4.3`；3-step `STAGE_TRACE_PASS` 36 records、50-step fixed-CFL stability 三路 PASS；25-step paired `c30695a7`→`b980b326` HIP `5961.450325→4794.353067 ms`，ratio `1.243432x`；正式作业 `COMPLETED/0:0` |
| 2026-09-24 | 完成 1D `EulerBackend` → shared `EulerDomainBackend` 的 CPU/HIP 薄 adapter，并增加 direct/shared stateful paired benchmark；收紧 backend kind/device identity，adapter 不持有第二份 CPU/GPU state | Kunshan CPU root build、adapter `3/3`、normal/strict 各 `5/5`；standalone HIP GoogleTest/CTest `15/15`；root HIP build/smoke、GoogleTest `15/15`、hardware CTest `16/16`；NoTrace、FullTrace、stats、device identity、lifecycle reuse 全通过。`nx=1048576,steps=100,repeats=5,warmup=2` 的 direct/shared advance 为 `420.163270/420.126780 ms`，ratio `0.999913`，最终误差 `0`、launch/sync 完全一致；所有成功作业均 `COMPLETED/0:0` |
| 2026-09-23 | host lifecycle/boundary metadata 优化：跨 RK stage 复用 `UNsInvFlux` 的 `qf1/qf2/invflux`；HIP reconstruction 稳定保存 boundary operation host buffer，静态 operation 只在拓扑/host pointer 变化时 H2D；`NsCalcGamaT` 改用连续 field 指针访问 | Kunshan CPU normal/strict 五算例各 `5/5`，最终作业 workload 与 scheduler 均 `COMPLETED/0:0`；m6 同 basis `steps=3,warmup=1,repeats=3` 的 host-reuse HIP mean `2286.562 ms`，boundary-operation cache 版本 `1968.529 ms`，最终源码版本（含 boundary-face extent invalidation）HIP mean `1855.713 ms`、legacy mean `4834.289 ms`、单次作业 ratio `2.6051x`；跨作业 node/启动波动明显，未作为稳定加速结论；3-step HIP breakdown 的 `boundary_gamma_inner` 仍约 `73 ms`，下一步转向 boundary/ghost host loop 或 device update |
| 2026-09-23 | `CALC_BOUNDARY` 细粒度 profiling 与 inviscid boundary guard 复测：把 `LOAD_Q/CALC_TIME_STEP/LOAD_RESIDUALS/UPDATE_RESIDUALS/CALC_LHS/UPDATE_FLOWFIELD/CALC_BOUNDARY` 及 `NsCalcBoundary` 的 gamma/viscosity/BC 子项纳入 opt-in profiler；`vismodel=0` 跳过无用的 `CalcLaminarViscosity` | Kunshan root OneFLOW 增量 build/smoke、3-step 36-record accuracy、25-step `warmup=1,repeats=2` legacy CPU 8-rank/HIP 1-rank 与 3-step boundary substage profile 均 workload `exit_code=0`；guard 前后 HIP mean `81.288 s→81.247 s`，无可分辨性能收益；HIP `CALC_BOUNDARY` 约 `72.393 s`（`89.1%`），3-step `boundary_bc≈9422 ms` 明显主导；CPU normal/strict 五算例各 `5/5`，最大 absolute difference 分别 `4.970574442764598e-10` 与 `1.1072414686508214e-17`；不更新正式性能报告，下一步转向 device boundary/ghost update |
| 2026-09-23 | 收紧 backend-neutral HIP reconstruction view contract，并实现受限 residual→LHS→primitive state-update seam：state-owned residual 在 `CALC_LHS` 设备缩放，`UPDATE_FLOWFIELD` 设备完成五方程 primitive↔conserved 更新；NoTrace 不回传 residual，更新后的内部 `q` 暂回传 host 供 boundary/下一 stage 使用 | 本地 Release 增量构建 `oneflow_euler_domain_contract_test`；直接 GoogleTest `11/11`；`ULhs.cpp`、`UNsUpdate.cpp`、`UNsInvFlux.cpp` CPU 语法检查通过。Kunshan DTK 26.04 / `gfx906` / `dcu:1` root configure/build、HIP smoke、GoogleTest `9/9`、hardware CTest `10/10` 通过；新增 2-cell/2-ghost/3-face/5-equation state-update contract，device primitive update 最大误差 `1.388e-17`，HIP CTest smoke `1/1`（测试本体 `0.14 s`）；3-step stage `36` 条记录通过，legacy→CPU batch 最大绝对差 `2.2826185386293218e-13`、legacy→HIP batch `3.3528735343679728e-13`；fixed-CFL `0.01` 50-step 的 legacy/CPU batch/HIP batch 均 `exit_code=0`、`diagnostic_hits=0`、`PASS`，scheduler 与 workload 均 `COMPLETED/0:0`。未做正式 timing，不更新正式性能报告。 |
| 2026-09-23 | 同一昆山作业内做无 trace 的长步数完整 workload sweep，检查短运行是否掩盖 HIP 增益 | `test/m6wingroe_sa`、固定 CFL `0.01`、CPU 8 ranks、HIP 1 rank、每个 steps 1 warmup + 2 repeats；3/10/25/50 steps 的 legacy→HIP wall-clock 分别为 `1.2788x/1.3071x/1.3113x/1.3124x`，四组 workload exit 均为 `0`，作业 `COMPLETED/0:0`。结论：3→10 steps 有启动/初始化摊薄收益，10→50 steps 已基本平台化在约 `1.31x`；这是单次作业内探索性 sweep，不更新正式性能报告。 |
| 2026-09-22 | 完成 3D 主 solver P0 ownership/reconstruction contract scaffolding，并将 production gradient storage 接入 `Ns3DDeviceState` | 本地只做静态检查；Kunshan CPU normal/strict 各 `5/5` 且 scheduler `COMPLETED/0:0`；DTK 26.04 / `gfx906` / `dcu:1` root HIP configure/build、smoke、GoogleTest `9/9`、hardware CTest `10/10` 通过。3-step workload 生成完整 trace/log，但 PMIX cleanup 未返回，未将该 revision 的 stage verifier/workload exit 记为通过；无性能 timing。
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
