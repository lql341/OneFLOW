# OneFLOW 开发待办与衔接（living document）

## 交接摘要与整体路线图

> **本文只放三样东西**：当前状态（§1、§3）、Phase A–F 台账与证据（§4）、下一轮任务（§5）。
> 逐日轮次记录不进本文（见文末「归档说明」）。
> 长期方案的取舍见 [`oneflow-development-plan.md`](oneflow-development-plan.md)；
> 环境、固定路径与坑见 [`../handoff/oneflow-current-handoff.md`](../handoff/oneflow-current-handoff.md)；
> 分支与门禁规则见仓库 `AGENTS.md`；集群流程见 [`../../ci/kunshan/README.md`](../../ci/kunshan/README.md)
> 与 [`../runbooks/oneflow-kunshan-workspace-runbook.md`](../runbooks/oneflow-kunshan-workspace-runbook.md)。
>
> 编号约定：一律用 Phase A–F 的 id（`A1`、`B3`、`D1`…），不要再引入第二套字母表。

### 0. 一句话现状

受限无粘 3D 路径（5 方程 / Lax–Friedrichs / `ilim=0` / 稳态显式 RK / 1 rank 1 zone）
的 **q 已设备权威**：状态更新不再回写 host，边界 ghost、墙面状态、输出按需回拉。
8 zone / 8 rank / 1 DCU、1000 步：上一会话 paired 点是 31.64 ms/step，不能当作本轮 A 侧。
本轮同节点 paired（50 步 warmup、2 个正式 repeat）把 gradient host 镜像从每 stage 收到最后 stage：
**29.86 → 26.72 ms/step（−3.14 ms，−10.5%）**。单 rank 分项里 `gradient_mirror` 从 3.20 降到 0.93 ms/step。
功能线已通过 **PR #227** 进入 upstream `master`；工作区 runbook 已通过 **PR #238**
进入 upstream。

**当前（2026-10-31 收尾）**：`dev` 已合入 upstream `638276499`，CPU 五算例
**normal 5/5 + strict 5/5（绝对 1e-15 判据未变）**。DCU 侧唯一确定性红灯
`HIP.MainSolverFluxOneCall` 已修：烟测夹具仍按整个 face 步长填 `bcQ`，而生产路径
从 `1c470414a` 起只上传 `nEquations*nBoundaryFaces` 的连续缓冲。昆山 1 DCU 上
`OneFLOWHipSmoke` 的 boundary contract 为 PASS（ghost/reconstruction/generation 均为 0），
进程退出码 0。本次没有重跑 CPU 五算例：改动不进入 CPU 残差路径。

### 1. 状态快照

| 项 | 值 |
|---|---|
| refs | `master` = `origin/master` = `upstream/master` = `68b1e5f6a`（2026-11-08 三端同步，PR #253）；`dev` 已合并该上游 HEAD，领先 upstream 441 个提交（其中非 merge 切片略多于 375）；上游本次带来 Workbench/vencat 参考实现与测试发现契约 |
| upstream PR | #159 / #160 / #203 / #227 / #238 / #253 全部 MERGED（#227 = 3D HIP 功能线；#238 = 昆山工作区 runbook；#253 = upstream 同步与收口） |
| 集群口径 | Kunshan DTK 26.04 / `gfx906` / 1 DCU；CPU 基线在 `kshcnormal`；**1 DCU 槽 = 8 核** |
| 性能基线 | 8 zone / 8 rank / 1 DCU，1000 步、paired：**14.62 ms/step**（2026-11-07 同节点交错 5 对中位；旧默认 15.53）。对 8 核 CPU `288.20` → **19.7×**（CPU 侧仍是 F1 报告的 2026-11-02 口径，未重测）。历史点：15.59（F1 报告）、31.64（2026-10-25，不可当 A 侧） |
| 门禁基线 | 3 步多区 trace `MULTIZONE_TRACE_PASS` + resident contract `24/0/0`；200 步跨后端 residual `2.331035e-18` / aero `0.0`；**CPU 五算例 normal 5/5、strict 5/5**（2026-11-08 复跑，strict 判据仍绝对 1e-15、基线 `validated_commit=7bd358c0d`）；`HIP.MainSolverFluxOneCall` 在 1 DCU 上 PASS；全量 GoogleTest 未重跑 |
| 加速器构建 | 可选、后端无关：`ONEFLOW_ACCEL_SDK_ROOT`（+ 可选 `_ARCH`/`_FLAGS`），种类由 `ONEFLOW_ENABLE_HIP/CUDA` 或前缀内容推断；一键 `scripts/build-accel.sh`。不请求则完全不生效（CPU/CI 不受影响） |
| 已知红灯 | 确定性 DCU 红灯已消除。`ActionMapReadFileTest` 仅在 `ctest -j16` 全量并发下偶发（测试隔离）。限制器设备化的数值偏差 `1.113e-06` 未过门禁（B2 在途，不是红灯而是未完成）。**多卡不是红灯**：1/2/4 卡 HIP 路径可用，按每 rank 固定 ≈8,000 cell 的弱扩展是 100% / 82% / 67%（§4.8b）；早期 16 zone/2 DCU 的 ≈2.6 s/step 是派生算例漏设 `time_integral=1`、跑成 host LU-SGS 的假象（handoff §5 坑 18）。公开多卡结论仍缺 ≥1M 合法分区 fixture（D1）。 |
| 能力边界 | `NS_SOLVER` / 5 方程 / **Lax–Friedrichs 或 Roe** / `ilim=0`（Barth 已实现但数值未过门禁，默认不放行）/ `vismodel=0` / 单网格层 / 稳态显式多阶段 RK / `nGhostCells==nBFaces`；其它配置自动回退 host，显式请求不支持项 fail-fast。**多 zone/rank 可通过配置检查**；多卡要求每 rank 固定 cell 数（§4.8b），1 rank 多 zone 更慢（§4.1 A1.8） |
| 尚未覆盖 | Venkat、粘性/RANS、LUSGS、multigrid、双时间；IO/restart/气动力/可视化仍在 host；多卡缺 ≥1M 合法分区 fixture（D1，外部依赖）。多卡只按每 rank 固定 cell 数报告（§4.8b），固定网格强扩展不用于跨卡结论 |

### 1.1 工作环境与固定路径（不要另建目录）

这一节是"agent 不要到处乱放"的唯一出处；路径一律写成可解析的形式，具体绝对路径
不写进仓库（发布边界）。接手时按 [`doc/handoff/oneflow-current-handoff.md`](../handoff/oneflow-current-handoff.md)
执行。

| 位置 | 路径 | 用途 | 规则 |
|---|---|---|---|
| 本地仓库 | 本 checkout 根（`git rev-parse --show-toplevel`） | 源码、文档、提交 | 只在 `dev` 提交；`master` 只读，与 upstream 同步；不建第二份 clone |
| 本地临时 | `/tmp` | 一次性脚本、打包、对比 | 不入库；结论写回文档 |
| 集群接入 | `$HOME/.agents/skills/scnet-hpc`（**OpenAPI 优先**，SSH 次之） | 查询、提交、上传、探测 | 走技能入口；不手搓 ssh/scp 循环，不静默切换后端 |
| 昆山工作区 | **`$HOME/oneflow/`（唯一）** | 见下方固定布局 | 不新建第二个工作区 |
| 源码 | `$HOME/oneflow/src/OneFLOW` + `src/REVISION` | 每轮 `rsync -a --delete` 覆盖 | 不建 `OneFLOW-<hash>` 目录 |
| 构建 | `$HOME/oneflow/build/app`（`build/port`、`build/platform` 预留） | 固定目标、增量编译 | 不建 `build/<jobid>`；**不要重新 configure** |
| fixture | `$HOME/oneflow/fixtures/m6-8zone-cpu8/` | 唯一合法分区算例 | 只读；派生副本用 `prep-case.sh` |
| 证据 | `$HOME/oneflow/evidence/<gate>/<jobid>/` | `gate` / `paired-ab` / `cpu-regression` | 只增不改；大二进制提取指标后可删 |
| 快照 | `$HOME/oneflow/snapshots/*.tgz` | 保留最近 5 个 | `cleanup.sh` 轮转 |
| 日志 | `$HOME/oneflow/logs/` | 作业 stdout/stderr | 保留 14 天，不写 `$HOME` |
| 弃用 | `$HOME/oneflow/_attic/<date>/` | 待删物 | 观察一轮后清（`cleanup.sh`，7 天） |

**禁止清单**：在 `$HOME` 根或 `softwares/projects/` 下新建 OneFLOW 目录/日志/一次性脚本；
按 job 号或轮次新建 build/run 目录；把绝对私有路径、主机名、账号、作业号写进提交文档；
每次会话新建 dated handoff；绕开技能直接改远端状态。

作业与环境常量（`mpirun` 启动、`scripts/env.sh`、HCA、`OMPI_MCA_coll`、cmake/DTK 路径、
1 DCU 槽 = 8 核、基准口径）见 runbook 与 living handoff，本节不重复。

### 2. 回顾（2026-08 → 2026-10-25）

| 时间 | 阶段 | 交付 | 关键证据 |
|---|---|---|---|
| 2026-08 | accelerator 框架与 CPU 基线 | `codes/accel` 抽象、CPU backend、五算例回归、昆山 CI | 框架交付报告；当时明确"只有框架，无数值 kernel" |
| 2026-09-13 | 两条线融合（A–E 层） | upstream 基线 + Phase 1–3 + EulerDomain/StateRegistry + CPU vertical slice | bridge 2.22e-16；根 CTest 163/163；五算例 5/5 |
| 2026-09-20~24 | 3D DCU vertical slice（F1–F3） | root HIP 编译/smoke/contract、3D m6 1-step/3-step trace、fixed-CFL 50 步、梯度切片 | trace 3e-15 级；50 步三路 PASS |
| 2026-10-09~19 | 多 zone 与 halo | METIS `Partition`、多 zone device state/boundary、GPU-aware MPI probe、policy 收口 | 1D 双节点 4 rank/4 DCU hash 一致；`1.39/1.31/1.19/1.13x` |
| 2026-10-20~23 | host 链优化 | cell-CSR gradient、packed gradient/q 交换、接口字段裁剪、壁面状态裁剪、ghost 打包 | 59.4 → 32.8 ms/step（跨会话） |
| 2026-10-24~25 | **device 权威 q（本轮）** | 契约增加 host-mirror currency、状态更新不 D2H、按需回拉、host 派生场按需重建 | 同机 paired 38.26 → 31.64 ms/step；PR #227 合并 |

性能演进（8 zone / 8 rank / 1 DCU，1000 步）：`59.4 → 38.3（同机旧行为）→ 31.6 ms/step`。
注意 2026-10-24 记录的 `32.8` 与 2026-10-25 的 `38.3` 不是同一节点，跨会话只比 paired 差。

### 3. 全链路审计（当前）

| 链路段 | 状态 | 备注 |
|---|---|---|
| 网格读入 / 分区 / wall distance | host | 只摊在启动阶段 |
| solver/state 创建、init q | host + 一次性 H2D | 8 zone 1000 步时占 3% 左右 |
| **时间推进调度（RK stage 循环）** | **host task 序列** | 每 stage 派发 + fence，是剩余固定开销的主因（§4 F7） |
| gradient / reconstruction / flux+residual / state update / 本地边界 / 本地时间步 | device | 每 stage 一批 kernel，q 不回写 |
| q 与 gradient 的 interface 交换 | device pack + host MPI | 每步 4 轮（3 stage + 1 post），§4 A2 |
| limiter / LUSGS / multigrid / 双时间 | limiter = device（Barth）/ 其余 host | Barth 的 φ 由设备 `ReconstructionLimiterMode::DeviceCell` 计算（主机侧重复的 `CalcLimiter()` 已删）；数值仍差 1.1e-6 未过门禁，默认 `ilim=0`；Venkat 未实现，LUSGS/multigrid/双时间回退 host |
| 粘性 / 湍流 / SA | host | 能力外，回退；旗舰算例正在此列 |
| 残差诊断 / 正性检查 | device 归约标量 | 单 zone 已闭环，多 solver 聚合待做 |
| restart / 可视化 / 气动力 / Cp | host | 按需 D2H（气动力只回拉边界相邻 cell） |
| 分区与 halo 拓扑 | host + 设备端打包 | DeviceDirect 在 packed layout 上未重测 |

### 4. 路线图与执行进度（Phase A–F）

编号与 [`oneflow-development-plan.md`](oneflow-development-plan.md) §2 完全一致：**A1–A4 / B1–B6 /
C1–C5 / D1–D4 / E1–E5 / F1–F4**。本文只记勾选与证据，阶段的取舍、准入门与退出门见计划。
不再另起一套字母表；早期文档里的 `F7.x`/`F8.x`/`F9.x`/`G4`/`H3` 等编号已在各项后就地标注对应关系。

规则：**只勾有证据的项**；父项只有在子项全勾时才勾；`[ ]` 表示未开始或未通过门禁。

#### 4.1 Phase A — 3D 设备驻留执行（1/4）

- [x] **A1 固定开销**（原 `F7.1`/`F7.2` 另加三项死拷贝；同节点 29.9 → 26.7 → 21.0 → 18.5 → 17.3 → 15.6 → **14.62** ms/step（A1.7c 收口）；F1 报告已发布）
  - [x] A1.1 梯度主机镜像只在最后一个 RK 阶段做。气动力与 Cp 在步末才读，中间 stage 的镜像没人看。同节点 paired 每 stage 29.703/30.017 → 最后 stage 26.660/26.779 ms/step（−10.5%）；单 rank `gradient_mirror` 3.20 → 0.93。`ONEFLOW_HIP_GRADIENT_MIRROR_ON_DEMAND=0` 恢复每 stage
  - [x] A1.2 RK 基线状态与残差基底不再从主机搬。第一次只删 `LOAD_Q`/`LOAD_RESIDUALS` 的 host 写入、没补设备侧基线，残差在上一 stage 上继续叠加，200 步跨后端拓到 `5.52e-4`；把“不填 host 槽位”和“设备残差归零”绑成同一个谓词（`DeviceOwnsExplicitRkBookkeeping()`）后对齐。paired 26.91 → 20.97 ms/step（−22.1%）。多重网格、粗网格、双时间、粘性仍填槽。`ONEFLOW_HIP_SKIP_HOST_RK_SLOT_COPY=0` 恢复
  - [x] A1.3 设备权威路径不再搬 host 历史层。`UPDATE_UNSTEADY_FLOW` 每步把 `q2←q1←q` 各做一次全场拷贝；双时间被 capability 排除，restart 写盘时又从设备场重建，这些层在推进中无人读。跳过条件是 `NsDeviceAuthoritativeFlowEnabled()` + 显式定常 NS，`ONEFLOW_HIP_SKIP_HOST_HISTORY_SHIFT=0` 恢复。同节点 paired 21.50 → 18.50 ms/step（−14%）；restart auth-on `5.03e-15`、auth-off `4.84e-15` 均 PASS，trace `24/0/0`，200 步跨后端 `2.331035e-18`/`0.0`，CPU normal/strict 5/5
  - [x] A1.5 设备单网格路径不再写 host 残差账本。`ZERO_RESIDUALS`（命名残差清零）与 `STORE_RHS`（拷进 `FIELD_RHS`）只服务多重网格缺陷与 host 显式更新；设备单网格路径两者都不读，却是每步两次全场 host 写。判据统一为 `HostResidualBookkeepingNeeded()`（多重网格配置或设备不拥有账本时保留）。8 rank 门禁 3 发 `17.27 / 17.25 / 18.14 ms/step`，trace `24/0/0`、200 步 XB `2.331035e-18`/`0.0`、CPU normal/strict 5/5
  - [x] A1.6 分项归因修正。任务级 profile 里 `POST_PROCESS` 的 `mean_ms` 曾显示每步 7.7 ms，实为**周期性可视化输出**摊到每一步的均值：总 2193 ms / 300 步，把 `nVisualSave` 设成不输出后该任务降到 0.055 ms/step。因此不能把它当作每步固定开销，也不能据此设计优化。教训：`mean_ms` 必须和 `count × 发生结构` 一起看。
  - [x] A1.7 每步固定基数的边界测量（2026-11-04）。**关通信**、`rk_stage=1`、8 rank/1 DCU 下仍有 `8.93 ms/step`；该基数与 cell 数、zone 数、rank 数都基本无关（1/2/4 zone per rank 的 1-stage 分别为 8.93 / 7.61 / 8.65）。同时测得：关闭通信只省 **2.2 ms**（23.97→14.28 首发、16.51→14.34 热态），所以“通信主导”不成立，且 A2 的收益上限就是这 2.2 ms。这条把 A2/A3/A4 的目标从“通信”改成“每步基数”。**注意：本条的 8.93 由 A1.7b 修正过**——100 步/300 步会把 1.4–1.6 s 启动摊进 ms/step；按 1000 步斜率基数 ≈5.5 ms，归因见 A1.7b/A1.7c
  - [x] A1.7b 固定基数的第一次实测（2026-11-07，昆山 1 DCU / 8 rank / 8 zone，m6wing，1000 步）：
      `rk_stage=3` 15.58 ms/step，与 F1 报告的 15.59 同一口径复现（合入 upstream `f2fc6ea2e` 后）；
      `rk_stage=1`（关界面通信）6.88 ms/step，所以每多一个 stage 的边际成本约 **4.35 ms**。
      **短基准会把它算错**：同一配置每次启动约 **1.4–1.6 s**（网格/分区/首次分配），
      100 步口径会把启动摊成 +14 ms/step、200 步约 +7 ms/step，A1.7 原来的 8.93 属于这一类读数，
      下次引用基数必须写步数并做启动扣除。**队列 drain 不是基数**：
      `ONEFLOW_HIP_DEFER_SYNC=1` 在 1000 步只把 15.58 降到 15.25（−2.1%），1 stage 时 6.88→6.39。
      结论：基数 ≈ **5.5 ms/step**（1 stage，已扣启动），下一步要用 `macro_*` 分段把它归到具体段
  - [x] A1.7c 第一版端到端优化（2026-11-07，1 DCU / 8 rank / 1000 步，同节点交错 3 发）：
      基线 **15.57**（15.50/15.57/15.62）→ 三个开关全开 **14.55**（14.54/14.68/14.55），
      **−6.5%**，样本无重叠。门禁全绿：trace `24/0/0` + `max_abs_diff=0`、200 步 XB PASS。
      改动：残差缩放并入状态更新入口（`ONEFLOW_HIP_FUSE_SCALE_UPDATE`）、非物理标志留在设备上
      逐步检查（`ONEFLOW_HIP_DEFER_STATUS`，每步少 4 次 4 字节阻塞 D2H）、drain 只在需要回读处
      （`ONEFLOW_HIP_DEFER_SYNC`）、inflow 值不变时不再每 stage 重传（原来它成了延迟队列的阻塞点，
      `bnd_static_upload` 曾显示 4.2 ms/step）。**三个开关已改为默认开**（`=0` 回退），
      另加 `ONEFLOW_HIP_MAX_LAUNCHES_PER_DRAIN`（默认 32）防止多 rank 下队列溢出 abort。
      收口复跑（2026-11-08）：gate 三发 14.75 / 15.08 / 15.78（该节点，跨度 1.0），
      trace `24/0/0`、XB PASS、CPU 五算例 5/5 + 5/5；性能口径仍以同节点 paired 的 14.62 为准
  - [x] A1.8 多 zone/rank 的可行性（2026-11-04）。`Zone::CheckZoneRankCount` 只查 `nZones % nProc`，所以 16 zone / 8 rank 能通过配置检查；但运行期死在 `HIP gradient face pack has no resident gradient`——该 rank 还没有推进过那个 zone。修法：设备侧梯度打包在缺 resident gradient 时**回退主机镜像**（此时主机场是权威初值），不再抛异常。修后 `zpr=2/4` 能跑完，但**更慢**：16.45（1 zone/rank）→ 17.62（2）→ 25.24（4）。结论：**“一 rank 多 zone”不是解法**，它不减少 halo，反而按 zone 重复每步工作
  - [x] A1.4 基准 warmup 规则。首个 `hipMalloc` 与冷 case 目录会污染短基准：同一配置冷跑 `27.0`、热跑 `18.5–21.1 ms/step`。规则落在交接 §5，不是代码改动
- [ ] **A2 每步一次界面交换**（原 `F7.3`，**已降级、不排期**；现行方案见 §5.2 A2′）。**两处实测已缩小它的价值，动手前先重估**：(1) “删掉第 4 轮”被门禁否掉——跳过 post 轮后 200 步跨后端从 `2.33e-18` 拓到 `5.54e-7`（aero `4.0e-5`），那一轮承载 legacy 的一步滞后；(2) 单卡上关掉**全部**界面通信只省 **2.2 ms/step**，所以把 4 轮合成 1 轮的理论上限约 1.1 ms。多卡场景下的接口占比见 §5.2 A2′（32 zone/4 卡 `interface_mpi` 3.05 ms/step，占 29%）
- [ ] **A3 stage kernel 融合**（原 `F7.4`）：gradient → reconstruction → flux → lhs → update → boundary 在设备内串联。stage 边际成本以 A1.7b 的 1000 步斜率 **4.35 ms/stage** 为准（旧的 3.29/1.88/1.20 是短基准读数）；融合收益按“每步往返次数 × 单次开销”估（§5.2 A3：每 stage 6 次 → 4 次，约 −4 ms/step）
- [ ] **A4 HIP `EulerDomainBackend::Advance`**（原 `F7.5`）：RK 三阶段循环进 backend，只在宏步边界回主机（输出、门控）。设备基态留在设备上，不要把 A1.2 跳过的 host 槽位拷回来。**目标 9–10 ms/step**（按 14.62 起点与 A1.9 的“剩余差距是结构性”结论重估；旧的 12–14 是 A1 收口前估算）

#### 4.2 Phase B — 数值能力（0/6，退出门见 B-exit）

- [x] **B1** 无粘通量 scheme：**Roe 已设备化并过单面契约**；HLLC/AUSM 未开始，
    需要时按同一路径另开子项；LF 保留作 oracle
  - [x] **B1.0** scheme 透传：`nscom.ischeme` 经 `DeviceFluxScheme()` 显式映射进内核，
    替掉原先硬编码的 `1`。**命名陷阱保留作记录**：内核私有编号 `1`=Rusanov、`2`=Roe，
    与求解器的 `ISCHEME_ROE = 1`、`ISCHEME_LAX_FRIEDRICHSS = 5` 是两套编号，映射函数
    就是为了不让两边按名字互相推断；未实现的 scheme 抛 `invalid_argument`，
    capability 增加 `roeScheme` 位后接受 Roe
  - [x] **B1.1** 设备侧 Roe：`HipFluxBackend.hip` 的 `RoeFaceFlux` 按 `codes/ns` 的
    `NsInvFluxPolicy` 逐项对齐（Roe 平均态、Harten 熵修正、绝对特征值与耗散波系数、
    法向跳跃与耦合项），由 `scheme == 2` 分支调用；LF 分支保持逐位不变
  - [x] **B1.2** 单点数值契约：`oneflow_device_roe_contract_test` 3/3；单 zone 200 步
    `res_max_rel = 0.0`。**未记录**：3 步多区 trace 与 CPU 五算例的 Roe 口径还没跑过，
    在补这两条之前 Roe 只能算"单 zone 已验证"，不能当生产结论
- [ ] **B2** limiter：Barth / Venkat（cell 邻域极值），解除 `ilim=0`
  - 已落地：设备侧 Barth 因子的**契约测试** `oneflow_device_limiter_contract_test`
    （5/5，覆盖范围内、超上界、低于下界、反向法向、斜面），参考实现是主机
    `BarthFunction` + `ReconstructFaceIncrement` + `FaceSideDirectionDot`
  - **关键结论（2026-11-02）**：**φ 不能在主机算**。设备权威 q 下主机梯度镜像只覆盖
    边界相邻单元、且只在有读者时才更新；用它算 φ 得到陈旧因子。实测 `ilim=1`
    200 步残差偏离 `8.53e-05`，逐 step 累积
  - **已实现（2026-11-03/04）**：设备侧 Barth φ —— 每 cell 一个线程、5 个方程在寄存器、
    邻接表走两遍（邻域极值 + 逐面取 min）；`ReconstructionLimiterMode::DeviceCell`
    显式声明"因子由后端算"；重构按主机语义 `min(φ_l, φ_r)` 施加。同时**删掉主机侧
    重复的 `CalcLimiter()`**（每 stage 270 ms）改为只填中性值
  - **当前数值状态**：`ilim=0` CPU vs HIP `0.0`（逐位一致）；`ilim=1` 200 步差
    **`1.113e-06`**，未过 1e-12 门禁。已排除：面上混合规则、ghost φ、主干预。
    **下一个怀疑点是主机 `LimField::CalcFaceValue` 的正性回退**：主机按面用
    `ckfun`（`NsCheckFunction`）整组丢弃受限值，设备是逐方程判定候选后回退单元值，
    两者不等价，量级与实测偏差相符
  - **端到端（单 zone，1 核 vs 1 DCU，200 步，关输出）**：`ilim=0` 1621.9 vs 24.25
    ms/step（66.9×）；`ilim=1` 2617.3 vs 63.67（41.1×）。设备限制器仍带来约
    39 ms/step 代价，其中 `recon_limiter_kernel` 仅 0.03 ms/调用，其余未定位
    （怀疑是每 stage 的 `FillNeutralLimiterField` 全场填充）。注意这是 **1 核 vs 1 卡**
    口径，不能与 8:1 槽的 15.6 ms/step 直接比
  - **B2.0 逐点比对已完成（2026-11-08，单 zone / 1 步 / `ilim=1`）**：新增
    `ONEFLOW_UNS_LIMITER_DUMP=<dir>`，把主机与设备的 φ 和面值各写一份二进制后再比：
    - `phi`：1,474,560 个值中 **6,665 个不一致（0.45%）**，形态是主机 0 / 设备 1（幅度 1.0），
      即"该单元是否取极值"判定不同；
    - `qf1/qf2`：各 4,481,280 个值中 **40,562 / 38,415 个不一致（≈0.9%）**，
      最大差 7.49e-01，集中在 eq=1（u 分量），且差异面与 φ 差异单元对应。
    结论：**不是整体公式错，而是少量单元/面的判定不同**，与"整组丢弃 vs 逐方程回退"或
    "邻域 stencil 不同（边界/ghost 面是否参与 min/max）"两类假设都相容。
  - 待做（B2.1）：按差异面清单核对 Barth 邻域构造（设备 `BarthLimiterKernel` 走 cell→face CSR
    且按 `boundaryMask`/`boundaryOperation` 跳过边界；主机 `Limiter::CalcLimiterScalar` 的邻域
    需逐项对照），再决定是补整组判据还是对齐 stencil；之后过 200 步跨后端与 CPU 五算例，
    再压 `FillNeutralLimiterField`，最后才是 Venkat（`ilim=2`）
- [ ] **B3** 层流粘性：面粘性通量、温度梯度、壁面热流。先做 E1（去 globals）
- [ ] **B4** 湍流：SA（1 方程）；SST 可选项，不单列
- [ ] **B5** 隐式：LU-SGS 设备化（下/上扫、`dq` 交换、与 halo 的依赖），再评估双时间。依赖 A4
- [ ] **B6** 多重网格：restrict/prolong 设备化，或明确声明只加速 finest grid。依赖 A4
- [ ] **B-exit**（原 `F9.1`）`test/cfd/m6wingroe_sa`（Roe + LU-SGS + SA）在设备上通过同一组门禁。这是 Phase B 的退出门，做完 B1–B6 不等于 B 结束

#### 4.3 Phase C — 真实算例与 IO（0/5）

- [ ] **C1** IO / restart：设备侧输出，或整场一次性 D2H + 异步写盘。设备权威 q 下的 restart 历史层已在 2026-11-01 修好（见 §5 第 3 项），但长跑与多输出仍缺量化
- [ ] **C2** 气动力 / Cp / 热流：设备侧面量归约，或维持边界单元级小回拉并量化代价
- [ ] **C3** 残差与收敛判据的设备归约，扩到多 zone、多 solver 聚合（目前只有单 zone 证据）
- [ ] **C4** 内存预算：记录每 100 万 cell 的常驻字节数，给出单卡可解规模
- [ ] **C5** 定常/非定常真实工况回归，与 `test/suites` 对齐，形成“设备算例集”

#### 4.4 Phase D — 多卡与多节点（0/4）

- [ ] **D1** 合法 fixture（原 `G4`）：≥1M cell 且带 `.skl` 的分区网格，或修 `GenerateLink`（`FaceTopo::SetNewFace2Node`）。外部依赖，越早启动越好
- [ ] **D2** 3D 设备 halo：GPU-aware MPI 与 DeviceDirect 在 packed layout 上重测（原 `G5`；1D 已验证 `1.39/1.31/1.19/1.13x`）
- [ ] **D3** rank/zone 映射与负载（原 `G6`）：1 rank = 1 zone 是否长期成立；需要 zone 切分时 state 与 halo 语义全变
- [ ] **D4** 评测口径：固定 per-rank cell 数、同一 repeats、1 DCU 槽 = 8 核，并说明 8:1 与 32:1 的合法性

#### 4.5 Phase E — 架构通用化（0/5；E1 已完成的部分见前置层）

- [ ] **E1** 热路径去 ambient globals（原 `H3`）：把 zone/grid/solver 身份显式传进 stage，先设备路径再主机
- [ ] **E2** capability 表驱动（原 `H2`）：solver × scheme × limiter × viscous × time-integral × grid-level × halo 一张表 + 单测。已有 `EulerInvFluxCapability` / `EulerRungeKuttaCapability` 两个枚举与稳定拒绝原因作起点
- [ ] **E3** `EulerDomain` 泛化到 NS（原 `H4`）：补源项、粘性、能量扩展与 view 元数据；1D 与主 solver 共用 `Advance` 语义并入这一项
- [ ] **E4** backend-kind 真实入口（原 `H5`）：清掉把 `IsAccelerator()` 当 HIP 用的地方，为 CUDA/Kokkos 留真实入口。构建期 SDK 接口已可选（`ONEFLOW_ACCEL_SDK_ROOT` + `--arch sm_80`），运行期入口仍是保留位，无 NVIDIA 机器实测
- [ ] **E5** 清理 toy/scalar 路径与失效文档（原 `H6`），保持 CPU oracle 单一

#### 4.6 Phase F — 产品化与发布（0/4）

- [x] **F1** 3D 正式性能报告（原 `G7`/`I1`）：同节点、同 repeats、paired。
      2026-11-02 发布，Markdown 与 HTML 同 stem：
      [`…-current.md`](../reports/performance/oneflow-3d-euler-performance-current.md) /
      [`…-current.html`](../reports/performance/oneflow-3d-euler-performance-current.html)。
      8 核 CPU 288.20 vs 1 DCU 15.59 ms/step（中位数，3 样本交替），**18.5×**；
      口径为 1000 步、50 步 warmup、关闭中间输出（输出摊薄实测 0.8 ms/step）
- [ ] **F2** CI 矩阵（原 `I2`）：CPU normal/strict、DCU 单卡、DCU 多卡、nightly 长跑；PR 分支自跑门禁
- [ ] **F3** 文档与能力矩阵（原 `I3`）：哪些 solver/scheme 在设备上已验证
- [ ] **F4** GUI / 示例算例 / 发行说明（原 `I4`）

#### 4.7 前置层（已完成，不计入阶段进度）

这些是加速主线开工前的地基。早期用 A–E 字母编号，和 Phase A–E 撞名，现在只留名字。

| 前置层 | 证据 |
|---|---|
| 运行基线 | `master`/`dev` 三端同步与分支模型走通；`sync-master.sh` macOS/Linux 双兼容；可选、后端无关的加速器构建接口（`cmake/OneFLOWAccelSDK.cmake` + `scripts/build-accel.sh`，昆山全新目录 configure 28 s） |
| standalone 1D 端口 | CPU/HIP lifecycle、NoTrace/FullTrace、单卡 HIP contract、1/2/4/8/64 rank 与 1/4/8 DCU 的 hash 与分解、应用级 `3.85–9.70x`（对 32 rank CPU）、4 卡/4M `25.55x` |
| accel substrate | `AccelRuntime`/`AccelBackend`/`DeviceBuffer`、`FluxBackend` 与 backend-neutral views、backend-kind 分派 |
| domain contract | `EulerDomain` views、`StateRegistry` 生命周期、`INIT_FLOWFIELD`/restart hook、`ResidentStateContract`（owner token、代际、ghost epoch、镜像新鲜度） |
| CPU 垂直切片 | CPU adapter、registry 生产化、RK seam 与回退、50 步输出级 oracle、逐面 trace、物理与守恒门禁 |
| DCU 受限路径 | root HIP 编译/烟测/契约、3D m6 精度门禁、fixed-CFL 50 步稳定、七段设备链、设备权威 q、多 zone 与 200 步跨后端 |

#### 4.8 顺序与插空规则

1. 已定：先把 Phase A 收到 ≤14 ms/step，再扩 Phase B。**当前项是 A3（stage 融合第一段）**；A2 已降级、不排期，多卡接口聚合见 §5.2 A2′。
2. B1（Roe）与 B2（limiter）是 B-exit 的前置；B3/B4 依赖它们；B5/B6 依赖 A4。
3. E1（去 globals）应在 B3 之前，最好也在 A4 之前。CUDA/Kokkos 当前不做，但不要把 HIP 类型写进 `EulerDomain` seam。
4. D1（fixture）是外部依赖，不阻塞 A/B。
5. F1 任何时点都能做（paired 口径已具备），建议 A 收口后立刻发一次。
6. D2/D3 依赖 D1，拿到合法 fixture 之前只做只读评估，不出数字。

#### 4.8b 并行扩展实测（强扩展 2026-11-08；弱扩展 CPU/DCU）

**CPU 强扩展实测（2026-11-08，`a02r4n11`，m6wing 294,912 cell，1–32 核）**

| cores | ms/step | 加速比 | 效率 | 每翻倍 |
|---:|---:|---:|---:|---:|
| 1 | 1580.32 | 1.00 | 100% | — |
| 2 | 892.54 | 1.77 | 88% | 1.77× |
| 4 | 474.06 | 3.33 | 83% | 1.88× |
| 8 | 279.32 | 5.66 | 71% | 1.70× |
| 16 | 142.46 | 11.09 | 69% | 1.96× |
| 32 | 73.70 | 21.44 | 67% | 1.93× |

拟合 `T ∝ n^-0.85`（六点误差 <5%），外推 512 核约 38% 效率。关通信对比
（`a02r4n14`，51 步两点法）：halo 占 8 核 30%、16 核 16%、32 核 **13%**。
32 核分项（`cpu32`）：`rhs_inviscid` 55.64（其中 gradient/reconstruction 各 20.98）、
界面约 11.5、其余约 6.8 → 每 cell 每 stage 约 0.76 µs（合理值 20–50 ns）。
**弱扩展阻塞已解除（2026-11-08）**：`codes/special/src/Transfinite.cpp` 的三维
transfinite 插值有两处复制粘贴错误——k 方向混合用了 j 向的边值（`dk*v0` 应为 `dk*w0`），
`uv` 角点项最后一项乘了 `ck` 应为 `cj`。这会把内部节点放到错误位置，单元数越大反转越多：


| Dim（每边节点） | 单元数 | 负体积单元 | 占比 |
|---:|---:|---:|---:|
| 3 | 8 | 0 | 0% |
| 9 | 512 | 95 | 18.6% |
| 17 | 4,096 | 921 | 22.5% |
| 33 | 32,768 | 7,734 | 23.6% |
| 65 | 262,144 | 62,816 | 24.0% |

`minimal_cube` 与 `two_connected_cubes` 两种布局在 Dim=33 下给出**完全相同**的 7,734，
说明问题在生成器的单元连接/体积约定（随规模放大才显形），不是布局方向写错。
修复（`172500ca1`）后重测：Dim=9/17/33/65 **负体积单元全部为 0**，可正常求解。

**弱扩展实测（2026-11-08，每核/每 rank 固定 ≈8,000 cell）**

注意：单核 cube 是单 zone，`nIFaces = 0`，不参与 halo 交换，因此 1 核点不能当弱扩展基线；
下面以 8 核点为基准。

| cores | 每边节点 | cells | CPU ms/step（3 发中位） | CPU µs/cell | 弱扩展效率 |
|---:|---:|---:|---:|---:|---:|
| 8 | 41 | 64,000 | 58.33 | 0.911 | 100% |
| 16 | 51 | 125,000 | 60.42 | 0.483 | 96.5% |
| 32 | 65 | 262,144 | 66.30 | 0.253 | **88.0%** |

DCU 同口径（同一套 cube 与分区，`e06r2n06`，1/2/4 卡 × 8 rank/卡）：

| ranks / DCU | cells | HIP ms/step（3 发中位） | HIP µs/cell | 弱扩展效率 | 对 CPU 加速 |
|---:|---:|---:|---:|---:|---:|
| 8 / 1 | 64,000 | 5.06 | 0.079 | 100% | 11.5× |
| 16 / 2 | 125,000 | 6.14 | 0.049 | 82.4% | 9.8× |
| 32 / 4 | 262,144 | 7.54 | 0.029 | **67.1%** | 8.8× |

读数：**工作量和核数同步放大时两边都能扩**（CPU 88%、DCU 67%），每 cell 成本持续下降
（CPU 0.911→0.253，DCU 0.079→0.029 µs）。DCU 的弱扩展效率低于 CPU，因为它的接口部分
（host staging + 4 轮同步往返）不随 cell 数下降；绝对性能上 DCU 仍快 8.8–11.5×。
单发噪声可达 3–4 倍（32 rank/4 卡三个样本 5.87 / 23.32 / 7.54），必须取中位并同作业内比较。
**但这组数字有一个未控制变量：每 rank 的 halo 并不相同，见下。**

**匹配性复核（2026-11-09，等块设计，CPU 8→128 rank）**

为了让每个 rank 拿到相同的 20³ 块，按 40³ / 40×40×80 / 40×40×160 / 80³ / 80×80×160
生成 cube 并切 8/16/32/64/128 zone（1/2/4 节点 × 32 核，1000 步 ×3，斜率扣启动）：

| ranks | cells/rank | nIFaces/rank（min..max，mean） | ms/step（中位斜率） | 弱扩展 |
|---:|---|---|---:|---:|
| 8 | 8000 | 1212..1494，**1324** | 57.9 | 100% |
| 16 | 7989..8010 | 1254..2214，**1647** | 58.4 | **99.1%** |
| 32 | 7977..8027 | 1281..2459，**1786** | 61.2 | 94.6% |
| 64 | 7983..8025 | 1339..3026，**2200** | 67.8 | 85.4% |
| 128 | 7980..8028 | 1285..3165，**2319** | 78.9 | 73.4% |

读数：**cell 平衡是 ±0.3%，但每 rank 的接口面数涨了 1.75×**——METIS 没有给出等块
（2×2×8 本应是 20³）。所以旧表里的 88% 混进了分区 halo 增长。

**D1a 结构化等块分区（2026-11-09 已实现）**：`partition_type = 2` 按 cell-count 分位切
a×b×c 块（`codes/partition/{include,src}/StructuredPartition.*`，9/9 单测）。同一套
cube 重测（1/2/4 节点，1000 步 ×3）：

| ranks | cells/rank | nIFaces/rank（min..max，mean） | 块排列 |
|---:|---|---:|---|
| 8 | 8000 | **1200..1200，1200** | 2×2×2 |
| 16 | 8000 | 1200..1600，1400 | 2×2×4 |
| 32 | 8000 | 1200..1600，1500 | 2×2×8 |
| 64 | 8000 | 1200..2400，1800 | 4×4×4 |
| 128 | 8000 | 1200..2400，1900 | 4×4×8 |

现在 cell 严格平衡、接口数完全由块的位置决定（角块 3 面=1200，边块 4 面=1600，
面内 5 面=2000，体块 6 面=2400），不再是分区器的随机结果。

**但弱扩展仍不能直接用 8 当基线**，两个原因：
1. **角块效应**：8 块全是角块（3 面=1200），域变大后内部块占比上升，每 rank halo 向
   2400 饱和；这是开放边界下弱扩展的固有几何效应，不是代码缺陷。
2. **睿频**：8/16 rank 只占节点一半核，频率高于 32 rank 全核；本作业 8→16 时间
   +22%，远大于 halo +17% 能解释的部分。
所以 **99–100% 只能用“整节点 + 同 halo”的配置来主张**（例如 32→64→128 或更长的
4×4×4→8×8×8），并且表里必须带 `nIFaces/rank`。本日的 64/128 三个样本跨度 21–31%
（集群同时有其他作业），时间斜率还不足以下结论；再用专门分配重测。

**固定 rank 的端到端收益（2026-11-09，32 rank / 1 节点，同作业 3 对交错，1000 步）**：
METIS 平均 `nIFaces` 1786 → 结构化 1500（−16%）；每步 **61.32 → 58.62 ms
（中位，−4.4%，3/3 对为正）**。样本：METIS 66.10 / 61.32 / 60.87，结构化
58.61 / 59.03 / 58.62。所以 D1a 不只是测量工具：固定 rank 下它直接省约 4%，
因为步长由 halo 最大的 rank 决定，而结构化块把最坏 rank 的 halo 从 2459 降到 1600。

DCU 侧同一套等块分区还未重跑；在它到位前，67% 只能作为“含 halo 增长”的混合读数。

#### 4.9 整体进度

| 阶段 | 勾选/总数 | 状态 |
|---|---|---|
| Phase A 设备驻留执行 | 1/4 | 🟨 A1 完成（**14.62 ms/step**，默认配置），A2/A3/A4 在途 |
| Phase B 数值能力 | 1/6（另有 B-exit） | 🟨 B1（设备 Roe）完成（HLLC/AUSM 未开始，Roe 只到单 zone 证据）；B2 在途，差 1.1e-6 |
| Phase C 真实算例与 IO | 0/5 | ⬜ |
| Phase D 多卡 | 0/4 | ⬜ 卡在 D1；弱扩展 1/2/4 卡已测（§4.8b），DCU 67% vs CPU 88% |
| Phase E 架构通用化 | 0/5 | ⬜ |
| Phase F 产品化 | 1/4 | 🟨 F1 报告已发布 |
| **阶段合计** | **3/28** | 前置层完成；A1、B1、F1 收口 |
| 前置层 | 6/6 | ✅ 见 §4.7 |

### 5. 下一轮任务（按序，含验收）

> 本节只列"接下来一轮"的工作，用勾选表示状态；能力台账与证据长期放在 §4，不在这里加新能力项。
> 每轮结束就地更新勾选，不新增日期章节。

#### 5.1 P0 — 先把基线恢复到可引用

- [x] **G0 门禁复跑**（合入 upstream `f2fc6ea2e` 后）。2026-11-08 全绿：
      trace `24/0/0` + `max_abs_diff=0`、200 步 XB `2.331035e-18`/`aero 0.0`、
      CPU 五算例 normal 5/5 + strict 5/5、1000 步 rc=0
- [x] **A1.7+ 固定基数已归因**（详见 §4.1 A1.7b/A1.7c）。结论：短基准把 1.4–1.6 s 的启动
      摊进 ms/step，旧的 8.93 属于这类读数；按 1000 步斜率，每步成本 ≈ 同步往返次数 × 单次开销，
      与 cell 数无关。据此做的第一版改动见 A1.7c（默认生效，14.62 ms/step）
- [x] **A1.7d 默认开关复核**（2026-11-09，m6wing 8 zone / 8 rank / 1 DCU，1000 步、3 对交错）：
      `ONEFLOW_HIP_DEFER_SYNC` on/off = **13.220 / 13.651 ms/step（中位，−3.2%，3/3 对为正）**，
      默认（on）正确；同节点绝对值 13.2 低于 14.62，但跨节点/跨日不可直接比。
      分项（同作业，defer off 以得到干净的每段执行）：`gradient` 2.28、`reconstruction` 1.95、
      `flux_assembly` 1.83（设备 kernel 合计 4.46），接口非嵌套合计 ≈ 4.54（`interface_mpi` 2.19），
      `macro_calc_lhs` 0.57、`macro_time_step` 0.28、启动约 1.6/步（1000 步口径）

**已收口的前置修正**（稳定结论，数字见 handoff §4）

- [x] **扩展性口径**：强扩展的 67% 是"固定网格被过度切分"，不是代码不能扩。判据一律
      改成**每核/每 rank 固定 cell 数**（≈8,000），并按弱扩展效率报告；单 zone 算例
      `nIFaces = 0`，**不能**当弱扩展基线。
- [x] **网格生成**：`Transfinite.cpp` 的 3D 插值曾有两处复制粘贴错误（k 方向取错边值、
      `uv` 项系数错），导致网格放大后约 24% 单元负体积。已修，现在 `minimal_cube` 可
      任意放大到 65³（262,144 cell）并正常求解。
- [x] **分区配方**：`gridObj = 3` 写 `script/grid.txt`，`npartproc` 写 `script/part.txt`
      （后加载者生效）；求解算例必须补 `grid/bcRegionMap.txt`；`time_integral` 必须显式
      设 1。三条缺一都会伪装成"扩展性问题"或崩溃。完整配方见 runbook §1b。
- [x] **合入 upstream `68b1e5f6a` 后门禁复跑**：trace `24/0/0`、200 步 XB
      `2.331035e-18`/`aero 0.0`、CPU 五算例 normal 5/5 + strict 5/5。

#### 5.2 P1 — Phase A 收口（目标 ≤14 ms/step）

> **本轮顺序（2026-11-09 按实测调整）：A2′ 接口 → A3 融合 → A4。** 单卡分项显示接口
> ≈4.54 ms/step、设备 kernel ≈4.46 ms/step，各占约三分之一；A3 的可及收益按实测评到
> 0.4–1 ms（不再是 −4），所以先把接口放到第一项。

- [x] **A1.9 梯度 unpack 路径的两处重复工作**（2026-11-08）：ghost-face→cell 映射按
      (grid, ghost level) 缓存（`006f9ff39`+`6174b21bd`），unpack 的 ghost id 列表按指针复用
      上传（`bbecf8241`）。门禁全绿（trace `24/0/0`、XB `2.331035e-18`）。
      500 步分项：`gradient` 5.27 → 4.87、`swap_download` 5.04 → 4.55；
      但 1000 步墙钟在噪声内（14.61–15.56），**尚未换成端到端收益**。
      **2026-11-08 复核（1/2/4 卡，`e07r2n03`，1000 步 ×2）**：8 zone/1 卡 17.37/14.61，
      16/2 卡 10.64/10.76，32/4 卡 10.27/10.21 → 1→2 卡 1.37×、2→4 卡 1.04×，
      扩展性没有变化。分项 `swap_download`：8 zone 4.45、16 zone 2.60、32 zone 1.09。
      结论修正：之前的"4.5 ms 是每 stage 的 D2H"判断不成立——设备 unpack 路径本身不拷贝，
      这个计时器是**延迟队列在下一个阻塞点的等待**，随 rank 数下降正好说明它承接的是
      排队工作量。所以"安全桶 5–6 ms"的估计过高，实际只兑现约 0.4 ms（分项），
      端到端在噪声内。剩下的差距是结构性的（A4 把循环下到设备），不是缓存能解决的
- [ ] **A3 stage 融合，第一段**：`gradient → reconstruction → flux` 串进设备，stage 内不再有 host 同步。
      **2026-11-09 实测修正（1000 步、8 rank/1 DCU、`ONEFLOW_HIP_DEFER_SYNC=0` 分项）**：
      `gradient` 2.28（kernel 1.48）、`reconstruction` 1.95（kernel 1.16）、`flux_assembly` 1.83，
      合计 **4.46 ms/step**；融合能拿回的主要是 qf1/qf2 全场往返（约 0.4 ms/step）和 1–2 次 launch，
      **不再是原估的 −4 ms**（那个估算基于 A1.7c 之前的“每次往返 0.4 ms”，延迟 drain 已把它拿走）。
      验收：trace `24/0/0`、XB `2.331035e-18`/`0.0`、同节点交错 paired 有可测收益
- [ ] **A2′ 接口交换**：4 轮、每轮每 rank 约 3 条 zone-pair 小消息，32 zone/4 卡时
      `interface_mpi` 3.05 ms/step（占每步 29%），是**延迟主导**而不是带宽主导。
      **2026-11-09 单卡实测（同上作业）**：`interface_mpi` 2.19、`post_upload` 0.89
      （含 `swap_upload` 0.88）、`post_download` 0.50、`interface_pack` 0.56、
      `interface_unpack` 0.42，非嵌套合计 ≈ **4.54 ms/step（33%）**，与设备 kernel
      4.46（3 stage 合计）持平——**当前最大的两个可改项之一**。
      2026-11-08 已验证两条路：
      - **GPU-aware 设备直发不可用**：`ci/kunshan/gpu-aware-mpi-probe.cpp` 在 2 rank/2 DCU
        单节点 **PASS**（device→device 点对点），但求解器开
        `ONEFLOW_ENABLE_UNS_GPU_AWARE_MPI=1 ONEFLOW_GPU_AWARE_MPI_CONFIRMED=1` 后，
        8 rank 与 32 rank 都在 `libgalaxyhip`/`hsa-runtime64` 里 abort（signal 6，
        rc=134）。探针不足以放行求解器路径。
      - 所以剩下的可行方向是**减少消息数**（把每个 rank 的多个 zone-pair 聚合成一条消息，
        或用 allgather 一次取全 halo），而不是换传输方式
      - **轮次 4→3 已再次验证不可行**：`ONEFLOW_SKIP_POST_INTERFACE=1` 后
        200 步 XB `5.540033e-07`、aero `4.0e-05`（fail），但 1000 步从 14.62 降到
        **13.73**（−0.9 ms）。即第 4 轮买 0.9 ms，却承载着 legacy 的一步滞后，
        要削它必须找到"同一滞后、少一次往返"的等价写法，不能直接删
- [ ] **A4 HIP `EulerDomainBackend::Advance`**：RK 阶段循环进 backend，只在宏步边界回主机。
      **前置已满足**：A1.7b/A1.7c 已把基数归到同步往返与 host 编排上，因此 A4 按结构性
      下沉估算（目标 9–10 ms/step，见 §4.1 A4）
- **A2 已降级，不排期**：上限只有约 1.1 ms（实测关全部界面通信仅省 2.2 ms/step），§4.1 A2 保留原证据

#### 5.3 P1′ — Phase B 并行（与 A 不抢同一段代码）

- [ ] **B2 收口**：按 B2.0 的差异清单（φ 0.45%、面值 0.9%）核对 Barth 邻域 stencil，
      定位 `1.113e-06`；若确认是主机 `ckfun` 的整体丢弃语义，设备侧补同样的按面判定。
      过 200 步跨后端与 CPU 五算例后，再压 `FillNeutralLimiterField`（当前 `ilim=1` 仍多约 39 ms/step）
      验收：`ilim=1` 200 步 XB ≤`1e-12` + CPU 五算例；之后才讨论放开默认
- [ ] **B1.2 补证据**：设备 Roe 目前只有单 zone 证据（契约 3/3、200 步 `res_max_rel = 0.0`），
      补 3 步多区 trace 与 CPU 五算例的 Roe 口径，否则 B1 不能写进 PR 结论
- [ ] **E1 热路径去 ambient globals**（§4.5 E1）：先设备后主机；**B3 的前置**，早点做可以避免 globals 扩散
- [ ] **B3 层流粘性**：面粘性通量、温度梯度、壁面热流（依赖 E1）
      验收：同 §4 的数值门禁 + 粘性算例 trace

#### 5.4 P2 — 外部依赖与后续

- [x] **D1a 结构化等块分区**（2026-11-09）：`partition_type = 2`，按 cell-count 分位切
      a×b×c 块（`StructuredPartition.*`，9/9 单测）。8→128 rank 每 rank `nCells` 严格
      8000，`nIFaces` 由块位置决定（1200/1600/…/2400），不再随 METIS 随机波动。
      验收前半（cell 相同）已达成；**后半（用它重跑 CPU/DCU 弱扩展）待专门分配**：
      本日样本跨度 21–31%，无法下结论，见 §4.8b
- [ ] **D1b 等 halo 弱扩展**：只在整节点配置间比（32→64→128，或 4×4×4→8×8×8），
      表里带 `nIFaces/rank` 与频率说明；验收：给出“扣除 halo 后的每 cell 计算时间”是常数
- [ ] **D1 合法 fixture**：≥1M cell 带 `.skl` 的分区网格（或修 `GenerateLink`）。
      外部依赖，越早启动越好；拿到之前 D2/D3 只做只读评估、不出数字。
      2026-11-08：分区配方与 manifest 校验已跑通（runbook §1b，16/32 zone 均 PASS），
      1/2/4 卡弱扩展 100% / 82% / 67%（§4.8b）。仍缺 ≥1M 的合法分区网格，
      公开多卡结论继续挂起（D1）
- [ ] **B4 SA**（依赖 B3）
- [ ] **B5 LU-SGS**（依赖 A4）
- [ ] **B6 multigrid**（依赖 A4）
- [ ] **B-exit**：`test/cfd/m6wingroe_sa`（Roe + LU-SGS + SA）在设备上过同一组门禁；
      B1–B5 全绿才可勾，这是旗舰算例离开 CPU 的唯一途径

C1–C5、D2–D4、E2–E5、F2–F4 均未启动，定义见 §4，不在本轮排期。

**不要做**：靠加卡提升（§4.8b 已说明）；"一 rank 多 zone"（A1.8 实测更慢）；
继续端到端盲试（曾因读数噪声或归因错误返工三次）。

### 6. 关键开关与脚本

| 用途 | 位置 |
|---|---|
| 后端选择 | `ONEFLOW_ACCEL_BACKEND=CPU|HIP` |
| 设备权威 q（默认开） | `ONEFLOW_HIP_DEVICE_AUTHORITATIVE_Q=0` 回退旧行为；`ONEFLOW_HIP_GHOST_Q_MIRROR=1` 只把 ghost q 的 D2H 加回（bisect） |
| 其它性能开关（默认全开） | `ONEFLOW_HIP_SKIP_HOST_INTERFACE`、`ONEFLOW_HIP_GRADIENT_MIRROR_ON_DEMAND`（默认只在最后 RK stage 镜像；`=0` 恢复每 stage）、`ONEFLOW_HIP_SKIP_HOST_RK_SLOT_COPY`（默认跳过设备显式路径上的 `LOAD_Q`/`LOAD_RESIDUALS` 槽位拷贝；`=0` 恢复）、`ONEFLOW_HIP_CACHE_INTERFACE_HANDSHAKE`、`ONEFLOW_HIP_SKIP_HOST_FIELD_PACK`、`ONEFLOW_HIP_SKIP_UNUSED_INTERFACE`、`ONEFLOW_HIP_DEVICE_GRADIENT_UNPACK` |
| 本轮新增（默认全开，`=0` 回退） | `ONEFLOW_HIP_DEFER_SYNC`（drain 只在需要回读处）、`ONEFLOW_HIP_DEFER_STATUS`（非物理标志留设备，宏步末检查一次）、`ONEFLOW_HIP_FUSE_SCALE_UPDATE`（残差缩放并入状态更新入口） |
| 队列深度 | `ONEFLOW_HIP_MAX_LAUNCHES_PER_DRAIN`（默认 32；多 rank 下不设上限会在 `libgalaxyhip` 里 AQL 溢出 abort） |
| 诊断 | `ONEFLOW_STAGE_BREAKDOWN_FILE`、`ONEFLOW_UNS_STAGE_TRACE_FILE`、`ONEFLOW_UNS_HIP_RESIDENT_CONTRACT=1`、`ONEFLOW_UNS_HIP_STAGE_DIAGNOSTIC=1` |
| 多 zone fixture 制作 | runbook §1b：`script/grid.txt`（并列入 `control.txt`）→ `simutask=Grid` → 补 `grid/bcRegionMap.txt` → 导出/merge/verify manifest |
| 门禁脚本（仓库内） | `ci/kunshan/f2-main-solver-{multizone-trace,cross-backend-residual,stability}.sh`、`f2-main-solver-paired-performance.slurm` |
| 标准作业脚本（集群固定工作区） | `scripts/{env,rank,build-app,prep-case,gate,paired-ab,cpu-regression,cleanup}`，规则见 [`runbooks/oneflow-kunshan-workspace-runbook.md`](../runbooks/oneflow-kunshan-workspace-runbook.md) |
| 加速器构建（仓库内，可选） | `scripts/build-accel.sh`；参数只有 `ONEFLOW_ACCEL_SDK_ROOT`（+ 可选 `_ARCH`/`_FLAGS`），种类自动推断；不设则 CPU/CI 不受影响 |
| 分支同步 | `scripts/sync-master.sh`（macOS/Linux 双兼容，支持 `--dry-run`） |
| 基准口径 | 1000 步、50 步 warmup、`nFieldSave/nResSave/nForceSave=100000`、CFL 0.01、`ilim=0`、`ischeme=5`、`time_integral=1`、`rk_stage=3`、`vismodel=0` |

### 7. 口径纪律（不遵守的结论一律作废）

1. 只比同节点、同 fixture、同 `steps/warmup/repeats`、同 rank/资源；`lifecycle_*_ms` 是 repeats 总和。
2. 不以 100 步或 kernel-only 作生产结论；1 DCU 槽 = 8 核。
3. 探索性单作业数据只进本文档或集群 artifacts，不进正式报告。
4. 任何性能改动先过：3 步多区 trace → 200 步跨后端 → CPU 五算例（normal/strict）→ 50 步稳定性。
5. 正式报告的 Markdown 与 HTML 必须同 stem 同步更新。
6. **数值参考数据的更新纪律**：先跑 strict 量清越界范围 → **只重生成越界算例**的
   `autotest/*.dat` → 用 `build_residual_db.py --absolute-tolerance 1e-15` 重建并
   `verify_residual_db.py` 校验 → 用**不变的** strict 判据复跑 5/5。判据本身不得为了
   通过而放宽；改成"精度优先"时只更新数据，不动判据。

---

## 归档说明

本文件在 2026-11-08 精简为 **living 大纲**：只保留当前状态（§1）、全链路审计（§3）、
Phase A–F 台账与证据（§4）、下一轮任务（§5）、开关与口径（§6–§7）。

**逐日的轮次记录、一次性实验数据与当时的判断已从本文件移除**，按 `doc/README.md`
的保留策略以 git 历史为档案（`git log -p doc/plans/oneflow-development-todo.md`）。
需要复用某轮结论时，从 §4 对应条目或 `handoff/oneflow-current-handoff.md` §5 的坑列表
进入；两者只保留**当前仍成立**的结论。
