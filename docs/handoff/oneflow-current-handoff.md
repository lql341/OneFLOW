# OneFLOW 交接（living）

> 2026-11-22 核对。工作分支 `dev`，当前接手项为 B.4.1 symmetry ghost 反射与 B.4.4 LU-SGS。
> 历史测量和旧章节从 Living TODO 的证据索引进入。

> **这是唯一的交接文档，就地更新，不再按日期新建。**
> 当前切片与验收勾选在 [`../plans/oneflow-development-todo.md`](../plans/oneflow-development-todo.md)；
> 长期阶段的取舍与里程碑在 [`../plans/oneflow-development-plan.md`](../plans/oneflow-development-plan.md)；
> 规则以仓库根 `AGENTS.md` 为准；集群操作流程见
> [`../runbooks/oneflow-kunshan-workspace-runbook.md`](../runbooks/oneflow-kunshan-workspace-runbook.md)。
>
> 本文只放"接手时立刻需要"的东西：**环境与固定路径、启动顺序、常量和坑、当前任务指针**。
> 当前验收以 Living TODO 为准，完整路线和任务编号以 Phase Plan 为准。

## 0. 60 秒上手

1. 读 `AGENTS.md`（分支模型、门禁顺序、口径纪律、工作区规则）。
2. 打开 **Living TODO 当前切片**。先验证 M6 粘性通量写回和气动力；
   H2 已在两节点两卡配置复验，A4 排在本轮正确性验收之后。
   **主线优先级只以 Living TODO 当前切片 为准**；本文只记“怎么跑”和“已经踩过的坑”，
   两边不一致时改本文，不要改 todo 的台账。
3. 要动集群先读 §1（固定路径）与 runbook：**OpenAPI 仍是结构化提交的默认后端**；
   SSH 已于 2026-11-19 续期恢复，可用于交互诊断、编译和传输。远端改动仍按固定
   工作区与可复现作业脚本执行，不要手搓 ssh/scp 循环。
4. 收尾：勾选 todo、证据落 `evidence/<gate>/<jobid>/`、更新本文 §4 与 §5；
   环境或路径变化同步 runbook 与 本文 §1。

## 1. 环境与固定路径（不要另建目录）

| 位置 | 路径 | 用途 | 规则 |
|---|---|---|---|
| 本地仓库 | 本 checkout 根（`git rev-parse --show-toplevel`） | 源码、文档、提交 | 只在 `dev` 上提交；`master` 只读、与 upstream 同步 |
| 本地临时 | `/tmp` | 一次性脚本、打包、比较 | 不入库；需要保留的结论写进文档 |
| 集群接入 | scnet-hpc 技能（按当前安装位置解析入口） | 查询、提交、上传、探测 | OpenAPI 为默认提交后端；SSH（`kseshell` / `zzeshell`）可用于交互诊断。不要手搓 ssh/scp 循环，也不要静默切换后端 |
| 昆山工作区 | **`$HOME/oneflow/`（唯一）** | `src/ build/ fixtures/ evidence/ snapshots/ scripts/ logs/ _attic/` | 固定布局，见下表与 runbook |
| 门禁证据 | `$HOME/oneflow/evidence/<category>/<jobid>/` | **6 个固定 category**：`gate` / `paired-ab` / `cpu-regression` / `probe` / `fixture` / `build` | 只增不改；不要按实验临时起目录名（规则见 runbook §1c）；`*.stage` 等大二进制提取指标后可删 |
| 源码快照 | `$HOME/oneflow/src/OneFLOW` + `src/REVISION` | 每轮覆盖 | 不建 `OneFLOW-<hash>` 目录 |
| 构建 | `$HOME/oneflow/build/app` | 唯一增量构建 | 日常不重新 configure；必要时按 runbook 的 SDK 接口在 DCU 分配内处理 |
| fixture | `$HOME/oneflow/fixtures/m6-8zone-cpu8/` | 唯一**登记**的分区算例（8 zone / 294,912 cell） | 只读；派生副本用 `prep-case.sh`。其它 zone 数用 runbook §1b 的配方现生成 |
| 快照归档 | `$HOME/oneflow/snapshots/*.tgz` | 保留最近 5 个 | `cleanup.sh` 轮转 |
| 日志 | `$HOME/oneflow/logs/` | 作业 stdout/stderr | 保留 14 天 |
| 弃用 | `$HOME/oneflow/_attic/<date>/` | 待删物 | 观察一轮后清（`cleanup.sh`，7 天） |

**集群接入现状（2026-11-19 更新）**：两个区域的 SSH 密钥已续期并在本机安装，
`ssh kseshell`（昆山 `login01`）与 `ssh zzeshell`（华中一区 `zz-login02`）均以 `luql`
登录成功；`scnet.py --backend ssh --cluster <c> doctor` 全项通过。密钥到期：
昆山 `2027-02-05`，华中一区 `2027-02-07`。旧的 2026-11-08 `Permission denied (publickey)`
记录已失效。

当前约束（区分后端）：

- **提交与查询**仍默认 OpenAPI：`submit --command` 的作业脚本路径是已审计的口径；
- OpenAPI 没有 `exec`，交互式 `make`、rsync 仍不可用；需要时用 SSH 直接做，
  或用 `git archive` tgz → `upload` → 作业内解包（两种都可，选一种并记录）；
- SSH 与 OpenAPI 不要混用同一作业状态：提交走哪边，查询/取消就走哪边；
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
| 环境 | 常规作业 source `scripts/env.sh` | MPI/UCX/DTK 必须显式进 PATH；rocprof 另需坑 31 的已验证 module 初始化 |
| HCA | `UCX_NET_DEVICES=mlx5_0:1`、`UCX_IB_DEVICES=mlx5_0` | 其它 HCA 的 `ibv_create_ah` 返回 EINVAL |
| MPI coll | `OMPI_MCA_coll=^hcoll,han` | 非均匀分布会崩在 `coll/han` |
| CMake | `/public/home/luql/apprepo/cmake/4.4.3/bin/cmake`（`env.sh` 已把它加入 `PATH`） | 旧的 `$HOME/tools/cmake-3.30.5` 已随家目录清理删除；模块里的 cmake 是死链 |
| DTK | `dtk-26.04`（显式加 `bin/llvm/bin/hip/lib` 等） | 不显式设会落到系统旧 ROCm |
| 资源口径 | 1 DCU 槽 = 8 核；CPU 基线在 `kshcnormal` | 跨队列比较必须换算到同一口径 |
| 基准 | 1000 步、50 步 warmup、`nFieldSave/nResSave/nForceSave=100000`、CFL 0.01 | 100 步或 kernel-only 不能作生产结论 |

## 3. 常用命令

**注意：提交默认走 OpenAPI 技能；SSH 已恢复（见 §1），可用于交互诊断和编译。**

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
  --work-dir <ws>/logs --queue kshdnormal \
  --nodes 1 --cpus 8 --dcus 1 --memory 27G --walltime 01:30:00 \
  --stdout <ws>/logs/of-gate-%j.out --stderr <ws>/logs/of-gate-%j.err
python3 $SKILL/scripts/scnet.py --region 11250 --details job <jobid>
python3 $SKILL/scripts/scnet.py --region 11250 logs --path <abs path> --lines 200
python3 $SKILL/scripts/scnet.py --region 11250 --timeout 180 upload <local tgz> <ws>/snapshots
```

门禁顺序固定：3 步多区 trace（含 resident contract）→ 200 步跨后端 → CPU 五算例
normal/strict → 50 步稳定性；性能改动再加 paired。任何一步失败都不算完成。

## 4. 当前任务指针（每次会话就地更新）

**2026-11-28 收口点**。`dev` HEAD `7b97d2b52`，已推送 `origin/dev`，工作树干净。
local/origin/upstream `master` 三端均为 `701234a71` 并已并入 `dev`。本轮 5 个 commit：
B.4.1 symmetry 算术共享、`-O3` 设备编译、LU-SGS profiler、门禁硬化、台账收口。

**已验证（1 zone / 1 rank）**：B.4.1 L1–L3 全过。
L3 full-dev 200 步 `129048798`：residual `3.2e-20`、aero `0`，三判据全 yes。
端到端 2.42× 是 1 核 / 1 DCU 口径，不可与 18.5×（8:1 无粘）混谈。
完整数值与证据索引见 [Living TODO](../plans/oneflow-development-todo.md)。

**8 zone / 8 rank（未完成）**：full-dev 数值 fail，已二分定位到 device 梯度链。
全 host `129082789` residual 0 pass；device flux on + 梯度 off `129083701` residual 3.4e-21 pass；
双 on `129056842` residual 2.0e-6 fail。强制下载假设被证伪（恶化到 1.4e-4）并已回退。
aero 3.6e-2 在纯 host 两臂同样存在，是分区累加差异，非 device 问题。
**下一步**：定位 `HipFluxBackend::CalcGradient` 的多区 ghost 梯度来源，修好后重跑 8 zone 20 步。

## 5. 已知坑与设计约束（只加有长期价值的）

### Device 路径与数值
- **双 opt-in**：`vismodel!=0` 时 `UseHipBatchAdapter`（`UNsInvFlux.cpp:899`）与 `UseHipGradient`
  （`:791`）各查一个独立开关；只开 batch 得「device flux + host 梯度」混合路径，
  `gradient_kernel=0`、`rhs_inviscid` 21→360 ms/step（`128983927`）。旗舰两个都要开。
- **三判据**：`DEVICE_FLUX_ON`（能力报告 `enabled=1`）、`DEVICE_GRADIENT_ON`（`gradient_kernel>0`）、
  `TIMING_USABLE`（两者皆 yes）。`rhs_inviscid` 仅参考，**禁止**单独判 device ON；
  `HIP kernel` 只计 launch，不含同步。工具：`ci/kunshan/parse-xb-device-path.py`（exit 0/2）。
- **LU-SGS 与 update_residuals 是兄弟阶段**：用 `lusgs / update_residuals` 算占比得 130%，
  分母错。单步顶层 = 残差 + LU-SGS 上下扫 + 边界 + 时间步 + 流场更新。
- **8 zone device 梯度**：见 §4 二分结论；这是当前唯一未解的数值问题。
- **单 zone aero 与多 zone aero 判据不同**：单 zone 可逐位比；8 zone 的 aero 受分区累加顺序影响，
  纯 host 两臂也差 3.6e-2，不能当 device 回归。

### 测量与工具链
- **每步残差/气动力输出毁掉多卡测量**：`nResSave=nForceSave=1` 时 `DumpFile()` 每步
  open/append/close，256 rank 时 `DUMP_RESIDUAL` 37.2 ms/步占 93%，随 rank 数增长。
  基准必须用 `prep-case.sh`（把三个 `*Save` 设 100000）。`sed` 改参数用 `[[:space:]]+` 并 grep 确认。
- **stage 分项表不是 GPU 时间**：host 计时器量的是入队+等待，`DEFER_SYNC` 开关能差 8 倍。
  `HipDrain` 每 rank 计数但 `hipDeviceSynchronize` 是设备级，8 rank 同卡时 per-rank
  `*_kernel` 横比无效（差 5.46 倍），要比较必须跑 `DEFER_SYNC=0` 对照。kernel 排期用设备侧计时，
  分项表只分 host 段。
- **GPU 占用率**：`/sys/class/drm/cardN/device/gpu_busy_percent`（`device/` 必须有），
  不能取全局最大值（会读到同节点其他租户）。`rocminfo` 不理会设备可见性变量，
  判断视图用 `hipGetDeviceCount` + `hipDeviceGetPCIBusId`。
- **rocprof 不可用**：v1 硬编码 `/opt/dtk-26.04/bin/rocminfo`，v2 `--kernel-trace` 在小内存 OOM
  且 `-d` 不落 CSV；补丁后仅 dispatch 计时可用。仅有 kernel 时间不能证明接近 roofline。
- **上游 `ProcessRunner.cpp` 曾假定 glibc ≥2.29**：昆山是 CentOS 7.6/glibc 2.17，
  已改版本判断；合并 upstream 后先在本集群编译 `OneFLOW_RuntimeObjects`。

### 集群作业
- **资源按节点算**：`max_cpus_per_node=32`、`max_dcus_per_node=4`、`--dcus` 每节点值、
  内存 ≤2 GB/core。`mpirun --host` 带槽位（`h1:32,h2:32`）。SSH 的 sbatch 脚本必须写
  `#SBATCH --gres=dcu:1` 且用 `--ntasks-per-node=8`（不是 `--cpus-per-task=8`）跑 8 rank。
- **SSH 密钥**：更换密钥后必须 `ssh-keygen -y -f <私钥> > <私钥>.pub` 重生成 `.pub`，
  否则 SSH offering 旧指纹→`Permission denied`。验证指纹用 `ssh-keygen -y -f` 管道，
  不要用 `ssh-keygen -lf <私钥>`（读 `.pub` 缓存，给假答案）。私钥权限 600。
  新密钥 2026-11-27 下载，到期 **2026-12-27**；备份在 `~/.ssh/backup-20261127/`。
- **shell 续行**：`VAR=... \` 中间插 `#` 注释会终止续行，env 全部丢失（作业 `129044654`
  计时全 0）。注释写在整条命令上方。
- **门禁数值 fail 也要出报告**：`f2-main-solver-cross-backend-residual.sh` 捕获 `numeric_rc`
  后仍写 `device-path.txt`；`set -e` 下直接 `sys.exit(1)` 会把最有诊断价值的产物藏掉。
- **OpenAPI 500 时走 SSH**：SSH 后端 `submit` 只接受 `--remote-path`（现成 sbatch 脚本），
  不支持 `--command`；`jobs` 不支持，用 `exec "squeue ..."`。
- **并发提交会踩构建目录**：多个作业同时 `of-sync-src.sh` 删/建 `build/app` 导致 `.o.d` 缺失，
  二分实验串行提交。

### 文档
- **文档根改名连带目录级路径**：`doc/`→`docs/` 时除 markdown 链接外还要改 CI 的 `cd`、
  `.readthedocs.yaml`、`docs/Makefile`、`tools/verification/contracts/**` 的 `references`；
  `scripts/check-doc-paths.py` 覆盖两类错误。
- **Living TODO 是活的 handoff**，phase plan 是长期计划，AI_CONTEXT 只做导航与当前状态；
  不要逐日堆积 slice 记录，证据放 `evidence/`，可复用经验放 `DEVELOPMENT_PRINCIPLES.md`。
## 6. 本文维护规则

- **不新建 dated handoff**：每次会话结束就地更新 §4（任务指针）和 §5（新坑）。
- §1/§2 只在环境或固定路径变化时改，并同步 `runbook` 与 本文 §1。
- 能力边界、性能数字、Phase A–F 进度不写进本文，避免与 todo 双份维护。

45. **bash `local` 同一语句内不能引用刚声明的变量（2026-11-15，一天内踩了四次）**：
    在 `set -u` 下，
    ```bash
    local tag="$1" dir="$EV/case-$tag"     # ✗ tag: unbound variable
    ```
    会失败——bash 把同一条 `local` 里的多个赋值视为**同时求值**，`$tag` 在展开时
    还未定义。必须分行：
    ```bash
    local tag="$1"
    local dir="$EV/case-$tag"               # ✓
    ```
    互不引用的同行赋值（`local a=$1 b=$2`）是安全的。
    **后果不只是报错**：如果这段代码在计时之后才执行，作业会以 FAILED 收场
    但**所有测量数据已经落盘且有效**——不要因为作业状态是 FAILED 就丢掉数据，
    先看 `summary.tsv` 是否完整；也不要因为数据有效就忽略这个 bug。
    排查顺序：`grep -n 'local .*=.*\$' script.sh` 找出所有同行引用。

46. **作业脚本里 `mpirun ./rank.sh` 要求 rank.sh 与 cwd 同目录（2026-11-15，踩了两次）**：
    `rank.sh` 在 `$SCRIPTS` 下，但 `mpirun` 的相对路径 `./rank.sh` 是相对**当前工作
    目录**解析的。做法照抄别处脚本时容易漏掉这行：
    ```bash
    EV="$ONEFLOW_ROOT/evidence/probe/${SLURM_JOB_ID:-manual}/<name>"
    mkdir -p "$EV"
    cp "$SCRIPTS/rank.sh" "$EV/rank.sh"; chmod +x "$EV/rank.sh"   # ← 必须
    # ... 之后 ( cd "$EV" && mpirun ... ./rank.sh ... )
    ```
    漏了会得到
    `prterun was unable to launch ... could not access an executable: ./rank.sh`，
    而且**只在 mpirun 那一步失败**，前面的检查全部通过，容易误判为"环境问题"。

47. **多个 agent 共享工作树时不要用 `git add -A`（2026-11-16，自己犯的）**：
    并行子 agent 都在同一个 checkout 里写文件，`git add -A` 会把**别人未提交的
    改动**扫进自己的提交。实际后果：一次 SA 数学优化的提交里混进了 156 行的
    头文件拆分重构，提交信息完全没提，后来者用 `git log` 查不到那次重构。
    规则：
    - 提交前先 `git status --short`，逐个文件确认归属；
    - 只 `git add <具体路径>`，不用 `-A` / `.`；
    - 如果发现已经混进去了且**已推送**，不要重写历史，改用说明性提交澄清，
      并把这次的归属记在提交信息里。
