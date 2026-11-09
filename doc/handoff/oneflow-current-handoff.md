# OneFLOW 交接（living）

> **这是唯一的交接文档，就地更新，不再按日期新建。**
> 当前状态与路线图勾选在 [`../plans/oneflow-development-todo.md`](../plans/oneflow-development-todo.md)；
> 长期阶段的取舍与里程碑在 [`../plans/oneflow-development-plan.md`](../plans/oneflow-development-plan.md)；
> 规则以仓库根 `AGENTS.md` 为准；集群操作流程见
> [`../runbooks/oneflow-kunshan-workspace-runbook.md`](../runbooks/oneflow-kunshan-workspace-runbook.md)。
>
> 本文只放"接手时立刻需要"的东西：**环境与固定路径、启动顺序、常量和坑、当前任务指针**。
> 能力边界、性能数字、Phase A–F 进度**不在这里重复**，一律以 todo 为准。

## 0. 60 秒上手

1. 读 `AGENTS.md`（分支模型、门禁顺序、口径纪律、工作区规则）。
2. 打开 todo **§5 下一轮任务**：当前主线是 **A3（stage 融合第一段）**，
   之后 A2′（接口消息聚合）、A4（RK 循环下沉）。A1 已收口在 14.62 ms/step。
   **主线优先级只以 todo §4/§5 为准**；本文只记“怎么跑”和“已经踩过的坑”，
   两边不一致时改本文，不要改 todo 的台账。
3. 要动集群先读 §1（固定路径）与 runbook：**SSH 已不可用，只能走 OpenAPI 技能**，
   因此一切远端操作必须写成作业脚本提交，不能在交互 shell 里敲。
4. 收尾：勾选 todo、证据落 `evidence/<gate>/<jobid>/`、更新本文 §4 与 §5；
   环境或路径变化同步 runbook 与 todo §1.1。

## 1. 环境与固定路径（不要另建目录）

| 位置 | 路径 | 用途 | 规则 |
|---|---|---|---|
| 本地仓库 | 本 checkout 根（`git rev-parse --show-toplevel`） | 源码、文档、提交 | 只在 `dev` 上提交；`master` 只读、与 upstream 同步 |
| 本地临时 | `/tmp` | 一次性脚本、打包、比较 | 不入库；需要保留的结论写进文档 |
| 集群接入 | `$HOME/.agents/skills/scnet-hpc`（OpenAPI 优先，SSH 次之） | 查询、提交、上传、探测 | 走技能提供的入口；不要手搓 ssh/scp 循环，也不要静默切换后端 |
| 昆山工作区 | **`$HOME/oneflow/`（唯一）** | `src/ build/ fixtures/ evidence/ snapshots/ scripts/ logs/ _attic/` | 固定布局，见下表与 runbook |
| 门禁证据 | `$HOME/oneflow/evidence/<category>/<jobid>/` | **6 个固定 category**：`gate` / `paired-ab` / `cpu-regression` / `probe` / `fixture` / `build` | 只增不改；不要按实验临时起目录名（规则见 runbook §1c）；`*.stage` 等大二进制提取指标后可删 |
| 源码快照 | `$HOME/oneflow/src/OneFLOW` + `src/REVISION` | 每轮覆盖 | 不建 `OneFLOW-<hash>` 目录 |
| 构建 | `$HOME/oneflow/build/app` | 唯一增量构建 | **不要重新 configure**（节点 DTK 的 hip CMake 校验会失败） |
| fixture | `$HOME/oneflow/fixtures/m6-8zone-cpu8/` | 唯一**登记**的分区算例（8 zone / 294,912 cell） | 只读；派生副本用 `prep-case.sh`。其它 zone 数用 runbook §1b 的配方现生成 |
| 快照归档 | `$HOME/oneflow/snapshots/*.tgz` | 保留最近 5 个 | `cleanup.sh` 轮转 |
| 日志 | `$HOME/oneflow/logs/` | 作业 stdout/stderr | 保留 14 天 |
| 弃用 | `$HOME/oneflow/_attic/<date>/` | 待删物 | 观察一轮后清（`cleanup.sh`，7 天） |

**集群接入现状（2026-11-08）**：`ssh kseshell` 返回 `Permission denied (publickey)`（密钥
失效或 IP 白名单变更），**只能用 OpenAPI**。由此带来的硬约束：

- 一切远端动作都要写成**作业脚本**用 `submit --command` 提交（建目录、解包、构建、清理都是）；
- OpenAPI 没有 `exec`，所以不能交互式 `make`，也不能 rsync；源码以
  `git archive` 打 tgz → `upload` 到 `snapshots/` → 作业内解包到 `src/OneFLOW`；
- 提交参数有内存/CPU 比例上限（`--memory` 过大直接被 sbatch 拒绝），4 核请用 ≤8G；
- 偶发 `Socket timed out` / `HTTP 502`：隔十几秒原样重试即可，不要当成作业失败。

**禁止清单**（写在这里就是为了不再重复）：

- 在 `$HOME` 根目录或 `softwares/projects/` 下新建 OneFLOW 目录、日志、一次性脚本；
- 按 job 号或轮次新建 build/run 目录（`build/<jobid>`、`runs/<name>-<jobid>`）；
- 在仓库里提交绝对私有路径、主机名、账号、作业号、原始 CI/Slurm 日志；
- 为一次会话新建 dated handoff 或多份并行计划（本文就地更新）；
- 直接跑 `scnet`/`ssh` 绕开技能去改远端状态；
- 按实验内容临时给证据目录或作业起名（`a17`/`ab2`/`attr` 这类）。作业名固定
  `of-<category>[-<purpose>]`，证据目录固定 `evidence/<category>/<jobid>/`，规则见 runbook §1c。

## 2. 昆山环境常量

| 项 | 值 | 原因 |
|---|---|---|
| 启动 | **只能 `mpirun --bind-to core --map-by slot`** | `srun` 下 MPI size=1，每个 rank 会解全部 zone |
| 环境 | 只 source `scripts/env.sh` | 批处理里的 `module load` 是空操作；MPI/UCX/DTK 必须显式进 PATH |
| HCA | `UCX_NET_DEVICES=mlx5_0:1`、`UCX_IB_DEVICES=mlx5_0` | 其它 HCA 的 `ibv_create_ah` 返回 EINVAL |
| MPI coll | `OMPI_MCA_coll=^hcoll,han` | 非均匀分布会崩在 `coll/han` |
| CMake | `/public/home/luql/apprepo/cmake/4.4.3/bin/cmake`（`env.sh` 已把它加入 `PATH`） | 旧的 `$HOME/tools/cmake-3.30.5` 已随家目录清理删除；模块里的 cmake 是死链 |
| DTK | `dtk-26.04`（显式加 `bin/llvm/bin/hip/lib` 等） | 不显式设会落到系统旧 ROCm |
| 资源口径 | 1 DCU 槽 = 8 核；CPU 基线在 `kshcnormal` | 跨队列比较必须换算到同一口径 |
| 基准 | 1000 步、50 步 warmup、`nFieldSave/nResSave/nForceSave=100000`、CFL 0.01 | 100 步或 kernel-only 不能作生产结论 |

## 3. 常用命令

**注意：SSH 不可用（见 §1），下面的 `sbatch` 只是作业脚本名；实际提交走 OpenAPI 技能。**

```bash
# 仓库侧
git -C <repo> status --short          # 提交前必须干净
# 集群侧脚本（都在固定工作区，由作业内执行）
source $HOME/oneflow/scripts/env.sh
$HOME/oneflow/scripts/build-app.sh                    # 增量编译（两步 make，见坑 12）
$HOME/oneflow/scripts/gate.slurm                      # 3 步 trace + 200 步 XB + 500 步分项 + 3x1000 步
$HOME/oneflow/scripts/paired-ab.slurm                 # 同二进制 paired A/B
$HOME/oneflow/scripts/cpu-regression.slurm            # CPU 五算例 normal/strict
$HOME/oneflow/scripts/cleanup.sh                      # 日志/快照/attic 轮转

# 提交（唯一方式，OpenAPI）
SKILL=$HOME/.agents/skills/scnet-hpc
python3 $SKILL/scripts/scnet.py --region 11250 submit --name of-gate \
  --command "bash /public/home/<user>/oneflow/scripts/gate.slurm" \
  --work-dir /public/home/<user>/oneflow --queue kshdnormal \
  --nodes 1 --cpus 8 --dcus 1 --memory 27G --walltime 01:30:00 \
  --stdout <ws>/logs/of-gate-%j.out --stderr <ws>/logs/of-gate-%j.err
python3 $SKILL/scripts/scnet.py --region 11250 --details job <jobid>
python3 $SKILL/scripts/scnet.py --region 11250 logs --path <abs path> --lines 200
python3 $SKILL/scripts/scnet.py --region 11250 --timeout 180 upload <local tgz> <ws>/snapshots
```

门禁顺序固定：3 步多区 trace（含 resident contract）→ 200 步跨后端 → CPU 五算例
normal/strict → 50 步稳定性；性能改动再加 paired。任何一步失败都不算完成。

## 4. 当前任务指针（每次会话就地更新）

**基线**：`master` = `origin/master` = `upstream/master` = `68b1e5f6a`（2026-11-08 三端同步，
PR #253）；`dev` 已合并该 HEAD。昆山工作区的 `src/OneFLOW` + `src/REVISION` 每轮覆盖。

**这一轮的结论（2026-11-08）**

- **单卡性能**：8 zone / 8 rank / 1 DCU / 1000 步，同节点交错 5 对中位
  **15.53 → 14.62 ms/step（−5.9%）**。三个开关默认生效：drain 只在回读处
  （`ONEFLOW_HIP_DEFER_SYNC`）、非物理标志留设备宏步末查一次（`ONEFLOW_HIP_DEFER_STATUS`）、
  残差缩放并入状态更新入口（`ONEFLOW_HIP_FUSE_SCALE_UPDATE`）；另有队列深度上限
  （`ONEFLOW_HIP_MAX_LAUNCHES_PER_DRAIN`，默认 32，多 rank 不设上限会在 libgalaxyhip 里 abort）。
- **门禁**：trace `24/0/0` + `max_abs_diff=0`、200 步 XB `2.331035e-18`/`aero 0.0`、
  CPU 五算例 normal 5/5 + strict 5/5。
- **扩展性（本轮最重要的修正）**：强扩展 67% 是**网格过分割**造成的假象——固定
  294,912 cell 时 32 核每 rank 只剩 9,216 cell。改成每核固定 ≈8,000 cell 后：
  CPU 8/16/32 核 58.33 / 60.42 / 66.30 ms（弱扩展 88%），
  DCU 8/16/32 rank（1/2/4 卡）5.06 / 6.14 / 7.54 ms（弱扩展 67%），
  同 per-rank 负载 DCU 比 CPU 快 **8.8–11.5×**。**不要再引用"2 卡以上无收益"**。
- **两个测量陷阱**：①单 zone cube `nIFaces = 0`，不能当弱扩展基线（曾误报 94%）；
  ②DCU 单发噪声可达 3–4 倍（32 rank/4 卡三发 5.87/23.32/7.54），必须取中位、同作业内比较。
- **修掉的 bug**：`codes/special/src/Transfinite.cpp` 3D 插值两处复制粘贴错误
  （k 方向用了 j 向边值；`uv` 末项乘 `ck` 应为 `cj`），修复前 65³ 网格 24% 单元负体积。
- **B2.0 已完成**：新增 `ONEFLOW_UNS_LIMITER_DUMP`，单 zone 1 步 `ilim=1` 逐点比对——
  φ 0.45% 单元不同（形态 0 vs 1）、面值 0.9% 不同（eq=1 为主），说明是**极值判定差异**，
  不是公式整体错。下一步核对设备 `BarthLimiterKernel` 的邻域是否排除了边界/界面面。

**下一步（按价值）**

1. **A3 stage 融合第一段**：`gradient → reconstruction → flux` 串进设备，stage 内不再有
   host 同步。单卡可及收益按“每步往返次数 × 单次开销”估：每 stage 6 次 → 4 次，
   约 −4 ms/step。验收：trace `24/0/0`、XB `2.331035e-18`/`0.0`、同节点交错 paired 可测。
2. **A2′ 接口聚合**：DCU 弱扩展 67% vs CPU 88%，差在接口；把每 rank 的多个 zone-pair
   小消息合成 1 条（GPU-aware 直发已验证**不可用**：探针 PASS 但求解器在 libgalaxyhip abort）。
3. **A4**：RK 循环下沉设备（结构性，单卡 14.6 → 目标 9–10）。
4. **B2.1（手段线，不占 A 线）**：按差异单元核对 Barth 邻域 stencil，解锁 `ilim=1`。
   stencil 只是手段，目标是把限制器变成可用能力；不要为了对齐 stencil 停住性能线。
5. **D1**：外部 ≥1M cell 合法分区网格。

**不要做**：单 zone 当弱扩展基线；用 <1000 步的短基准报 ms/step；跨节点比单发数字；
用 ad-hoc 分区做多卡结论（必须显式 RK + manifest 校验）。

## 5. 已知坑与设计约束（只加有长期价值的）

**测量与工具链（2026-11-08 新增，全部踩过）**

12. **`make OneFLOW/fast` 不会重编 `codes/` 下的源文件**：`<dir>/fast` 是 CMake 的"仅本目录"
    目标，只重链 `main/src/OneFlow.cpp`。必须两步：`make OneFLOW_RuntimeObjects` 再
    `make OneFLOW/fast`（`scripts/build-app.sh` 已修）。验证办法：`strings bin/OneFLOW |
    grep <新符号>`，没有就是没编进去。
13. **`StageProfiler` 会静默丢弃未登记的类别**：名字必须先出现在 `StageProfiler.cpp` 的
    `Totals()` 表里，否则计时被丢掉、TSV 里看起来"该段零成本"。现在 `Flush()` 会打印
    `dropped N timer records for unregistered category ...`，看到就补表。
14. **启动固定开销 1.4–1.6 s**：100 步口径把它摊成 +14 ms/step、200 步 +7。所有 ms/step
    必须写步数并扣启动（同一算例跑 1 步与 N 步取斜率）。旧的"8.93 ms 固定基数"就是这类读数。
15. **跨节点/单发噪声很大**：同一配置跨节点差 32%（13.12 vs 9.93），DCU 单发可达 3–4 倍
    （5.87 / 23.32 / 7.54）。**只比同作业内交错重复的中位**，跨节点单发一律不算数。
16. **延迟 drain 会让队列溢出**：`ONEFLOW_HIP_DEFER_SYNC=1` 下不设上限时，16/32 rank 会在
    `libgalaxyhip` 里 AQL 溢出 abort。`ONEFLOW_HIP_MAX_LAUNCHES_PER_DRAIN`（默认 32）必须保留。
17. **GPU-aware 探针通过 ≠ 求解器可用**：`gpu-aware-mpi-probe` 2 rank 单节点 PASS，但求解器
    开 `ONEFLOW_ENABLE_UNS_GPU_AWARE_MPI=1` 后 8/32 rank 都 abort。别再据此放行。

**集群作业与算例（同上）**

18. **派生算例必须显式设 `time_integral = 1`**：`test/cfd/m6wingroe_sa` 自带 `= 2`（LU-SGS），
    设备驻留链路只在 1 下启用；漏了会跑成 host LU-SGS 并伪装成"多卡扩展性差"（实测 2.6 s/step）。
    同时显式设 `rk_stage=3`、`ischeme=5`、`ilim=0`、`vismodel=0`，并删掉 `solver.txt` 的
    `TurbSolver`。
19. **分区参数分两处**：`gridObj = 3` 在 `script/grid.txt`，`npartproc`/`part_uns_file` 在
    `script/part.txt`（control.txt 里后者后加载，**覆盖**前者）。只在 grid.txt 写 `npartproc`
    会得到 `The number of partitions should be greater than 1`。
20. **求解算例必须带 `grid/bcRegionMap.txt`**：缺它会在 `UNsBcSolver::CalcBcRegion` 段错误，
    在分片导出阶段表现为 SIGSEGV。
21. **单 zone 算例 `nIFaces = 0`**，不做 halo 交换 —— 不能当弱扩展基线。弱扩展必须每核/每 rank
    固定 cell 数（≈8,000），且以多 zone 点为基准。
22. **分区作业的 case 目录要传绝对路径**：`... OneFLOW 0 .` 会静默什么都不写（rc=0 但没有输出
    文件），看起来像分区失败。
23. **cell 平衡 ≠ halo 平衡；扩展性用 `partition_type = 2`**（2026-11-09）：METIS 在等块
    设计下仍给出 1212–3165 的 per-rank `nIFaces`。新增结构化分区（D1a，`partition_type = 2`）
    按 cell-count 分位切 a×b×c，cell 严格等分、接口数由块位置决定（角块 1200、边块 1600、
    体块 2400）。固定 32 rank 同作业配对：平均 `nIFaces` 1786 → 1500，每步
    **61.32 → 58.62 ms（−4.4%，3/3 对为正）**。**扩展性表必须带 `nIFaces/rank`**。另两个坑：
    - **8 rank 不能当弱扩展基线**：只占节点一半核心，睿频更高；本作业 8→16 时间 +22%，
      远大于 halo +17%；
    - 开放边界下每 rank halo 会从角块的 1200 饱和到体块的 2400，弱扩展 99–100% 只能在
      整节点且同 halo 的配置间主张（D1b）。
    - **瓶颈已定位（2026-11-09）**：同一套等块分区、整节点 32/64/128 rank，
      `ONEFLOW_DIAG_NO_COMM=1` 后计算段完全持平（139–141，相对量），接口项
      13.5 → 22.9 → 27.9 ms（+106%），而面数只 +27% → 剩下的是跨节点延迟。
      下一步是 A2′（减少轮次/消息），不是继续调分区。

1. **设备权威不能从初始化就开始**：bound 状态的第一次 `FullField` 上传依赖 host 的
   ghost 区；跳过条件必须是 `HostMirrorStale(grid)`，否则第一步就偏离 CPU oracle。
2. **设备写 ghost 而未回写 host 时必须把镜像标脏**（`NoteDeviceGhostWrite(false)`），
   否则 contract 诊断"通过"而 host ghost 落后（1e-16 级），后续整场上传会把旧值灌回设备。
3. **`UploadBoundaryState` 不要按 generation 差异回灌 host 场**：设备权威时 owner
   generation 每 stage 前进，会把上一 stage 的旧场覆盖到设备。
4. **resident bcQ 复用已撤回**：solver 与 backend 两侧判据一旦不一致会上传未初始化
   buffer，3 步内 non-physical；收益只有 0.3–0.7 ms/step，不值。
5. **异常排查**：`Simulation::RunImpl` 的 `catch(...)` 会先 `MPI_Finalize` 再重抛，
   跨 rank 抛异常时 stderr 可能永远打不出来；临时在 catch 里先打印 `what()`。
6. **capability 边界**：只有 `NS_SOLVER / 5 方程 / (Lax–Friedrichs 或 Roe) / ilim∈{0, Barth} /
   vismodel=0 / 单 grid level / 稳态显式 RK / 1 rank 1 zone` 走设备；Barth 的数值还差
   `1.1e-6` 未过门禁，默认不放行；Venkat / HLLC / AUSM 未实现，显式请求 fail-fast。
   其它残缺配置回退 host。旗舰算例 `m6wingroe_sa` 是 `LU-SGS + Roe + SA`，仍在 host。
7. **短基准的假象**：`bnd_resize` 的首个 `hipMalloc` 会被平均进 ≤100 步的基准
   （10–15 ms/step 量级），任何基准先 warmup 至少一步。
8. **fixture**：唯一合法的 ≥8 zone 分区网格是 m6wing（294,912 cell）；多卡结论缺
   ≥1M 带 `.skl` 的网格，`partition_type=0`(GenerateLink) 仍失败。
9. **加速器构建用统一 SDK 接口，别再手拼**：运行时 `ONEFLOW_ACCEL_BACKEND`
   （AUTO/CPU/HIP/CUDA/KOKKOS）与构建期 `ONEFLOW_ACCEL_SDK_ROOT`（+ 可选 `_ARCH`/`_FLAGS`）
   是两条独立的轴；**SDK 种类不单独设参数**，由 `ONEFLOW_ENABLE_HIP/CUDA` 或前缀内容推断。
   `cmake/OneFLOWAccelSDK.cmake` 从前缀派生 `ROCM_PATH`、厂商包前缀、`amd_comgr`/
   `AMDDeviceLibs`/`hsa-runtime64` 与 `CMAKE_TRY_COMPILE_PLATFORM_VARIABLES`（CUDA 分支派生
   `CUDAToolkit_ROOT` 等）；不请求就完全不生效。一键入口：
   `scripts/build-accel.sh --arch gfx906 --flags "$ONEFLOW_DTK_HIP_FLAGS"`。
   原理与故障对照见 runbook §1.2 / §4。
10. **strict（1e-15）当前仍 4/5，且不止一处贡献者**：`plateuns2dslau2_34950_35000`
   已解决：声速**改用一次除法**（精度更高：与参考逐位一致率 100% vs 乘倒数 86.8%；
   护栅对物理密度无影响），并按“精度优先、可更新历史精度数据、strict 判据不变”重生成
   了**该算例**的参考曲线与基线条目（`validated_commit=7bd358c0d`）。现在
   **normal 5/5、strict 5/5（绝对 1e-15）**。其余四个算例参考未动、仍在判据内。已定位两处上游重构：
   `f7ae944c0`（双边声速）与 `6ca93f57c`（面谱半径/局部时间步）；探针数值非单调
   （中间提交会回到 `0.0`），说明多处末位扰动互相抵消。**已排除 fast-math**：
   当前代码在 `-Ofast` 与 `-Ofast -fno-fast-math` 下给出完全相同的 `1.778709301357682e-15`，
   旧代码两种设置都是 `0.0` → 差异是源码里运算结合顺序的确定性 ≤1 ulp 改变，不是编译器
   重排、也不含不确定性。要靠补公式还原不成立（与上游方向相反且易反复）；恢复 1e-15
   需按流程重生成基线或改判据形态，属数值口径决定，等 owner 拍板。
11. **gradient host 镜像默认只在最后 RK stage（A1.1）**：`ONEFLOW_HIP_GRADIENT_MIRROR_ON_DEMAND=0`
    才恢复每 stage 镜像。不要用上一会话的 31.64 ms/step 当本轮 A 侧；本轮同节点
    对照是 29.86 → 26.72。分项文件如果路径里没有 `%r`，8 个 rank 会写同一个文件，
    读到的是其中一个 rank，不是 8 rank 均值。
12. **`LOAD_Q`/`LOAD_RESIDUALS` 不是死写（A1.2）**：前者填 host RK 基态，后者把命名残差置成
    `-FIELD_RHS`，而 fused flux 核把它当**累加基底**上传。跳任务必须同时把基底改成
    设备侧归零，两边由 `DeviceOwnsExplicitRkBookkeeping()` 一个谓词控制
    （`codes/solver/include/ResidualBasePolicy.h`）。只改一边，读数会变成残差在上一
    stage 上累加：trace 仍过，200 步 XB 会拓到 `5.52e-4`。
13. **restart 历史层是 host 派生数据，dump 时必须从设备场重建**：restart 流写
    `q1`/`q2`，读盘用 level 1 复原当前场；设备链只回填当前 `q`，历史层会停在旧值
    （曾导致 auth-on dump 与 auth-off 差 41%）。现在 `Restart::Dump` 在
    `NsDeviceAuthoritativeFlowEnabled()` 为真时先回填 `q` 再 `InitUnsteady()` 重建。
    **判据要用所有权谓词，不要用刷新函数的返回值**：镜像可能已被别的读者刷新，
    那样会静默跳过修复（踩过一次，读数逐位不变）。
14. **`mean_ms` 会被周期性输出污染，必须连 `count` 一起读**：`POST_PROCESS` 曾显示
    每步 7.7 ms，实际总 2193 ms / 300 步、只在第 0/50/100… 步输出；把 `nVisualSave`
    设成不输出后降到 0.055 ms/step。误把均值当每步开销会选错优化方向（踩过一次）。
    `ONEFLOW_STAGE_BREAKDOWN_FILE` 的 `.tsv` 是嵌套计时，不能相加求总量。
15. **GPU-aware MPI 探针通过不代表该用**：同节点两 rank 设备指针直达 `pass=1`，但接到
    真实界面交换后是 48–50 ms/step，host staging 是 15.7。这条 MPI 配置下设备指针路径
    实际更贵。开关是 `ONEFLOW_ENABLE_UNS_GPU_AWARE_MPI` + `..._CONFIRMED`，保持关闭。
16. **每步固定基数与通信无关**：`ONEFLOW_DIAG_NO_COMM=1` 关掉全部界面通信，8 rank/1 DCU
    仍从 16.5 只降到 14.3（省 2.2 ms）；`rk_stage=1` 时仍要 8.93 ms。所以 A2（合并交换轮次）
    的理论上限就是 ~2.2 ms，不要按“通信 5 ms”的老估计算。这个基数不随 cell/zone/rank 变，
    是本阶段唯一未定位的大项。
17. **别用 2 个样本下结论**：8-zone 同配置两次样本 17.59 / 14.28（跨度 23%），
    16-zone 与 32-zone 的 7% 差异完全在噪声内。要比较配置，至少 5 个样本或做配对交替。
18. **`zones % ranks == 0` 只是配置层放行**：`Zone::CheckZoneRankCount` 只查整除，
    但多 zone/rank 在设备路径曾直接 Fatal（`HIP gradient face pack has no resident gradient`）。
    已改为“缺 resident gradient 时回退主机镜像”，能跑完；**但实测更慢**
    （16.45 → 17.62 → 25.24 ms/step 于 1/2/4 zone per rank），因为它不减少 halo、
    只按 zone 重复每步工作。不要把它当扩展手段。
19. **`StageProfiler` 的分类表尺寸必须等于初始化项数**：声明 76 槽只填 73 项时，
    剩余槽 `name == nullptr`，`Record()` 里 `strcmp` 解引用空指针 → 一开分项就段错误；
    而且**未注册的分类名会被静默丢弃**，据此判断“某段没执行”会得到错误结论（踩过一次）。
20. **上游把 runtime 源码移进 OBJECT 库后，HIP 要在 `tests/` 作用域再次 enable**：
    `CMAKE_HIP_LINK_EXECUTABLE` 是目录作用域变量，`codes/` 里 enable 不够，否则
    generate 阶段报错。修法见 `tests/CMakeLists.txt` 顶部。
21. **设备权威 q 下不要在主机算 φ**：主机梯度镜像只覆盖边界相邻单元、且只在有读者的
    步更新。用它算限制因子得到的是陈旧值，200 步残差偏 `8.53e-05`（100 步 `3.05e-05`，
    逐 step 累积）。限制器要么整段留 host（回退），要么把 φ 计算搬上设备，不能半搬。
    设备路径现在只在 `ilim=0` 时启用，`UseHipGradient` 里写明了这个原因。
22. **限制器在面上是"两侧取最小"，不是各用各的**：主机 `LimField::CalcFaceValue`
    用 `ModifyLimiter(phil, phir) = min(phil, phir)`（`ug.ireconface != 0`），
    并把同一个因子用在两侧重构上。设备若按左右各自的 φ 计算，200 步残差会差
    `8.5e-5`（实测）。改设备限制器时先对齐这条，再谈别的。
23. **post 步的 interface 轮不是冗余**：显式 RK 每个 stage 末尾都交换一次，看起来
    第 4 轮是重复；实测跳掉后 HIP 与 CPU oracle 在 200 步上偏离 `5.54e-7`（aero `4e-5`）。
    它参与 legacy 的一轮 lag。这类“看起来多余”的交换，改之前先跑 XB。
24. **冷启动第一次 1000 步不能当 paired 端点**：同一配置在冷 case 目录上首跑
    `27.0 ms/step`，同一作业第二次 `21.1`。跨作业/跨冷热的绝对值不可比，只有
    交替 interleaved 的 A/B 才算数；报数字要写清是不是热态。
25. **`HIP.MainSolverFluxOneCall` 的 boundary contract 已修，不要再按数值红灯查**：
   旧读数 `reconstruction 8.600e+00` 是第 4 个方程的期望 `bcQ`，设备侧读到 0；
   `stale-op 3.000e+00` 是旧夹具里的 `+3` 偏移，不是失败判据。原因是烟测仍按
   `nFaces` 步长填 `bcQ`，而 `1c470414a` 之后的上传按连续的 `nBoundaryFaces` 步长读。
   夹具必须与生产路径一样：固壁面是 face 数组前缀，`bcQ[equation] ==
   bcQ[0] + equation*nBoundaryFaces`。不满足时后端会在上传前拒绝，不再静默读错方程。
26. **`_attic` 里删构建树要留足时限**：本轮探测留下 ~3.6 GB（多份 `build/` + git
   bundle），`rm -rf` 在 15 min 的作业里被 scheduler 切断。清 `_attic` 用
   `scripts/cleanup.sh` 或单独申请 ≥30 min 的 CPU 作业，不要塞进验证作业尾部。
   另：`snapshots/` 已超过保留 5 个的策略，下次 `cleanup.sh` 轮转。
27. **`cmake` 必须在 `PATH` 上**：`env.sh` 只导出 `$CMAKE` 变量时，`cmake` 会落到系统
   2.8.12.2，`make` 自动 reconfigure 还会去用缓存里已删除的旧 cmake 路径。

## 6. 本文维护规则

- **不新建 dated handoff**：每次会话结束就地更新 §4（任务指针）和 §5（新坑）。
- §1/§2 只在环境或固定路径变化时改，并同步 `runbook` 与 todo §1.1。
- 能力边界、性能数字、Phase A–F 进度不写进本文，避免与 todo 双份维护。
