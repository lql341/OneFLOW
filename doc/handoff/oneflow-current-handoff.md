# OneFLOW 交接（living）

> **这是唯一的交接文档，就地更新，不再按日期新建。**
> 当前状态与路线图勾选在 [`../plans/oneflow-development-todo.md`](../plans/oneflow-development-todo.md)；
> 长期阶段的取舍与里程碑在 [`../plans/oneflow-development-plan.md`](../plans/oneflow-development-plan.md)；
> 规则以仓库根 `AGENTS.md` 为准；集群操作流程见
> [`../runbooks/oneflow-kunshan-workspace-runbook.md`](../runbooks/oneflow-kunshan-workspace-runbook.md)。
>
> 本文只放"接手时立刻需要"的东西：**环境与固定路径、启动顺序、常量和坑、当前任务指针**。
> 能力边界、性能数字、A–G 进度**不在这里重复**，一律以 todo 为准。

## 0. 60 秒上手

1. 读 `AGENTS.md`（分支模型、门禁顺序、口径纪律、工作区规则）。
2. 打开 todo 文首 **§4 路线图勾选树**，取第一个未勾的条目（当前是 `F7.1`）。
3. 要动集群就先读 runbook：工作区固定为 `$HOME/oneflow/`，只用那里的 8 个标准脚本。
4. 收尾：勾选 todo、把证据落进 `evidence/<gate>/<jobid>/`、更新本文 §4（任务指针）与
   §5（新坑）；环境或路径有变化时同步 runbook 与 todo §1.1。

## 1. 环境与固定路径（不要另建目录）

| 位置 | 路径 | 用途 | 规则 |
|---|---|---|---|
| 本地仓库 | 本 checkout 根（`git rev-parse --show-toplevel`） | 源码、文档、提交 | 只在 `dev` 上提交；`master` 只读、与 upstream 同步 |
| 本地临时 | `/tmp` | 一次性脚本、打包、比较 | 不入库；需要保留的结论写进文档 |
| 集群接入 | `$HOME/.agents/skills/scnet-hpc`（OpenAPI 优先，SSH 次之） | 查询、提交、上传、探测 | 走技能提供的入口；不要手搓 ssh/scp 循环，也不要静默切换后端 |
| 昆山工作区 | **`$HOME/oneflow/`（唯一）** | `src/ build/ fixtures/ evidence/ snapshots/ scripts/ logs/ _attic/` | 固定布局，见下表与 runbook |
| 门禁证据 | `$HOME/oneflow/evidence/<gate>/<jobid>/` | `gate` / `paired-ab` / `cpu-regression` | 只增不改；`*.stage` 等大二进制提取指标后可删 |
| 源码快照 | `$HOME/oneflow/src/OneFLOW` + `src/REVISION` | 每轮覆盖 | 不建 `OneFLOW-<hash>` 目录 |
| 构建 | `$HOME/oneflow/build/app` | 唯一增量构建 | **不要重新 configure**（节点 DTK 的 hip CMake 校验会失败） |
| fixture | `$HOME/oneflow/fixtures/m6-8zone-cpu8/` | 唯一合法分区算例 | 只读；派生副本用 `prep-case.sh` |
| 快照归档 | `$HOME/oneflow/snapshots/*.tgz` | 保留最近 5 个 | `cleanup.sh` 轮转 |
| 日志 | `$HOME/oneflow/logs/` | 作业 stdout/stderr | 保留 14 天 |
| 弃用 | `$HOME/oneflow/_attic/<date>/` | 待删物 | 观察一轮后清（`cleanup.sh`，7 天） |

**禁止清单**（写在这里就是为了不再重复）：

- 在 `$HOME` 根目录或 `softwares/projects/` 下新建 OneFLOW 目录、日志、一次性脚本；
- 按 job 号或轮次新建 build/run 目录（`build/<jobid>`、`runs/<name>-<jobid>`）；
- 在仓库里提交绝对私有路径、主机名、账号、作业号、原始 CI/Slurm 日志；
- 为一次会话新建 dated handoff 或多份并行计划（本文就地更新）；
- 直接跑 `scnet`/`ssh` 绕开技能去改远端状态。

## 2. 昆山环境常量

| 项 | 值 | 原因 |
|---|---|---|
| 启动 | **只能 `mpirun --bind-to core --map-by slot`** | `srun` 下 MPI size=1，每个 rank 会解全部 zone |
| 环境 | 只 source `scripts/env.sh` | 批处理里的 `module load` 是空操作；MPI/UCX/DTK 必须显式进 PATH |
| HCA | `UCX_NET_DEVICES=mlx5_0:1`、`UCX_IB_DEVICES=mlx5_0` | 其它 HCA 的 `ibv_create_ah` 返回 EINVAL |
| MPI coll | `OMPI_MCA_coll=^hcoll,han` | 非均匀分布会崩在 `coll/han` |
| CMake | `$HOME/tools/cmake-3.30.5-linux-x86_64/bin/cmake` | 模块里的 cmake 是指向死链的旧版本 |
| DTK | `dtk-26.04`（显式加 `bin/llvm/bin/hip/lib` 等） | 不显式设会落到系统旧 ROCm |
| 资源口径 | 1 DCU 槽 = 8 核；CPU 基线在 `kshcnormal` | 跨队列比较必须换算到同一口径 |
| 基准 | 1000 步、50 步 warmup、`nFieldSave/nResSave/nForceSave=100000`、CFL 0.01 | 100 步或 kernel-only 不能作生产结论 |

## 3. 常用命令

```bash
# 仓库侧
git -C <repo> status --short          # 提交前必须干净
# 集群侧（工作区固定）
source $HOME/oneflow/scripts/env.sh
$HOME/oneflow/scripts/build-app.sh                    # 增量编译
sbatch $HOME/oneflow/scripts/gate.slurm               # 3 步 trace + 200 步 XB + 500 步分项 + 3x1000 步
sbatch $HOME/oneflow/scripts/paired-ab.slurm          # 同二进制 paired A/B
sbatch $HOME/oneflow/scripts/cpu-regression.slurm     # CPU 五算例 normal/strict
$HOME/oneflow/scripts/cleanup.sh                      # 日志/快照/attic 轮转
```

门禁顺序固定：3 步多区 trace（含 resident contract）→ 200 步跨后端 → CPU 五算例
normal/strict → 50 步稳定性；性能改动再加 paired。任何一步失败都不算完成。

## 4. 当前任务指针（每次会话就地更新这三行）

- **阶段**：`F7` 固定开销收口 —— `F7.1` gradient 镜像只在最后 stage → `F7.2` 跳过
  `LOAD_Q`/`LOAD_RESIDUALS` 整场 host 拷贝 → `F7.3` 每步一次 halo → `F7.4` stage 融合 →
  `F7.5` HIP `Advance`。
- **最近交付**：device 权威 q（paired 38.26 → 31.64 ms/step，PR #227 已合并上游）。
- **决策待定**：完成定义（1M 单卡 vs 100M 多卡）、性能优先还是能力优先、隐式/多重网格
  是否设备化、zone 映射、CUDA/Kokkos 是否真做 —— 见 plan §6。

## 5. 已知坑与设计约束（只加有长期价值的）

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
6. **capability 边界**：只有 `NS_SOLVER / 5 方程 / Lax–Friedrichs / ilim=0 / vismodel=0 /
   单 grid level / 稳态显式 RK / 1 rank 1 zone` 走设备；其它配置回退 host，显式请求
   不支持的项 fail-fast。旗舰算例 `m6wingroe_sa` 是 `LU-SGS + Roe + SA`，仍在 host。
7. **短基准的假象**：`bnd_resize` 的首个 `hipMalloc` 会被平均进 ≤100 步的基准
   （10–15 ms/step 量级），任何基准先 warmup 至少一步。
8. **fixture**：唯一合法的 ≥8 zone 分区网格是 m6wing（294,912 cell）；多卡结论缺
   ≥1M 带 `.skl` 的网格，`partition_type=0`(GenerateLink) 仍失败。

## 6. 本文维护规则

- **不新建 dated handoff**：每次会话结束就地更新 §4（任务指针）和 §5（新坑）。
- §1/§2 只在环境或固定路径变化时改，并同步 `runbook` 与 todo §1.1。
- 能力边界、性能数字、A–G 进度不写进本文，避免与 todo 双份维护。
