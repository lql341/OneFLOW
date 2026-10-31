# 昆山工作区与作业规范（runbook）

> 适用范围：昆山 SCNet 集群上所有 OneFLOW 相关工作（CPU 分区 `kshcnormal`、
> DCU 分区 `kshdnormal`）。目标：**一个固定工作区、固定路径、固定作业脚本**，
> 不再出现"每次一个目录、按 job 号建 build、日志散落在 home 根目录"。

## 1. 固定路径

工作区根固定为集群上的 `$HOME/oneflow/`（仓库里写成 `<workspace>/`，绝对路径只出现
在集群侧工作区 README，不进仓库）。远端操作走 `$HOME/.agents/skills/scnet-hpc`
（OpenAPI 优先、SSH 次之），不要手搓 ssh/scp 循环。除既有的共享工具链前缀外，不得在
别处新建 OneFLOW 工作区：

- 旧的按日期命名的实验目录（`softwares/projects/OneFLOW_*`，2026-08/09）已在
  2026-10-25 删除——源码在 Git 历史，证据在 `doc/`，不要再按那种方式建目录。
- 家目录根下不要留 OneFLOW 日志或一次性脚本：日志进 `logs/`，一次性脚本进
  `_attic/<date>/`，可复用的提升为 `scripts/` 里的标准脚本。

```text
<workspace>/
├── src/OneFLOW/                当前源码快照（每轮覆盖，不建 OneFLOW-<hash>）
│   └── REVISION                repo / branch / commit
├── deps/metis-install/         自编译 METIS 5.0.1
├── build/app/                  root 生产构建（HIP+MPI+METIS），增量
├── build/port/                 预留：standalone 1D 端口 contract 构建
├── build/platform/             预留：CPU-only root 构建
├── fixtures/m6-8zone-cpu8/     唯一合法可分区 fixture（只读）
├── evidence/<gate>/<jobid>/    门禁证据：gate | paired-ab | cpu-regression
├── snapshots/*.tgz             源码快照，保留最近 5 个
├── scripts/                    env.sh + 标准作业脚本 + cleanup.sh
├── logs/                       作业 stdout/stderr，保留 14 天
└── _attic/<YYYYMMDD>/          待删物，观察一轮后清空
```

规则：

1. **源码**：把仓库快照覆盖到 `src/OneFLOW/`（`rsync -a --delete` 或解压覆盖），
   同步更新 `src/REVISION`。不要在 `src/` 下保留多个修订目录。
2. **构建**：`build/app` 是唯一生产（HIP）构建目录，按目标固定、增量编译。
   - 日常增量：`scripts/build-app.sh`（即 `make -C build/app -j<N> OneFLOW/fast`）。
   - 上游改动 `CMakeLists.txt` 后 `make` 会自动 reconfigure。**不要再手拼 cmake 变量**，
     统一走仓库里的加速器 SDK 接口（`cmake/OneFLOWAccelSDK.cmake` +
     `scripts/build-accel.sh`）；它把下面三条一次配齐：
     1. 作业要在 **DCU 节点**（`kshdnormal`，`dcu:1`；`/dev/kfd` 与
        `HIP_VISIBLE_DEVICES` 可见）——CPU 队列上没有设备，检测不到架构；
     2. `ROCM_PATH`：DTK 的 `hip-targets.cmake` 用
        `set(_IMPORT_PREFIX "${ROCM_PATH}/hip" "/opt/dtk/hip")` 定位
        `libgalaxyhip`，不设就落到 `/opt/rocm` → 报 “install faulty /
        not all the files it references”；
     3. `amd_comgr` / `AMDDeviceLibs` / `hsa-runtime64` 要传给 cmake，**且**用
        `CMAKE_TRY_COMPILE_PLATFORM_VARIABLES` 让 `enable_language(HIP)` 的
        try_compile 子工程继承，否则报 “Could not find amd_comgr”。

     ```bash
     source $ONEFLOW_ROOT/scripts/env.sh          # 提供 ROCM_PATH 与 ONEFLOW_DTK_HIP_FLAGS

     # 全新配置（例如预留目录）
     bash $ONEFLOW_SRC/scripts/build-accel.sh --sdk auto --sdk-root "$ROCM_PATH" \
       --arch gfx906 --build-dir $ONEFLOW_ROOT/build/platform_hip \
       --flags "$ONEFLOW_DTK_HIP_FLAGS" --jobs 8

     # 增量更新生产构建
     bash $ONEFLOW_SRC/scripts/build-accel.sh --sdk hip --sdk-root "$ROCM_PATH" \
       --arch gfx906 --build-dir $ONEFLOW_ROOT/build/app \
       --flags "$ONEFLOW_DTK_HIP_FLAGS" --jobs 8
     ```

     `--sdk auto` 会按前缀内容识别 `hip`/`cuda`；`--configure-only` 只配置不编译。
     CUDA 侧用同一接口：`--sdk cuda --sdk-root <CUDA Toolkit> --arch sm_80`。

   - CPU-only 备用构建（不涉及 HIP 时可用，可在 CPU 节点配置）：`build/platform`。
     CPU 门禁可以指向它：

     ```bash
     export ONEFLOW_CPU_BIN=$ONEFLOW_ROOT/build/platform/bin/OneFLOW
     bash $ONEFLOW_ROOT/scripts/cpu-regression.slurm
     ```

     `cpu-regression.slurm` 默认仍用 `$ONEFLOW_BIN`（`build/app`），只在显式设置
     `ONEFLOW_CPU_BIN` 时切换，并把 `system/` 拷到被测二进制同目录。
     实测两种构建对 CPU 五算例给出**完全相同的残差**，可互为对照。
3. **fixture**：`fixtures/m6-8zone-cpu8/` 只读。派生算例用
   `scripts/prep-case.sh <dest> <steps>` 写到 `evidence/.../<jobid>/` 或
   `$TMPDIR`，作业结束随手清掉。
4. **证据**：`evidence/<gate>/<jobid>/` 只增不改；`*.stage` 这类大二进制在
   记录完指标后可以删除，但 `result.txt`、rank breakdown、作业输出必须保留。
5. **日志**：Slurm 的 `-o/-e` 指向 `logs/`（见脚本头），不要再落到 `$HOME`；
   `scripts/cleanup.sh` 保留 14 天。
6. **废弃**：旧 build/run/源码快照不要就地留着，移入 `_attic/<date>/`，下一轮
   清空。

## 2. 作业规范

| 约定 | 值 |
|---|---|
| 作业名 | `of-gate` / `of-paired` / `of-cpureg`（与证据目录同名） |
| 启动方式 | **只能 `mpirun`**；`srun` 下 MPI size=1，每个 rank 会解全部 zone |
| 布局 | `--bind-to core --map-by slot`；8 ranks/卡用 `scripts/rank.sh` 映射 `HIP_VISIBLE_DEVICES` |
| 环境 | 只 source `scripts/env.sh`（MPI 5.0.11 / UCX 1.22.0 / gcc16 / DTK 26.04 / cmake 4.4.3；`env.sh` 会把 `$CMAKE` 的目录前置到 `PATH`）；批处理里的 `module load` 是空操作 |
| 网络 | `UCX_NET_DEVICES=mlx5_0:1`、`UCX_IB_DEVICES=mlx5_0`（其他 HCA 会 `ibv_create_ah` EINVAL） |
| MPI coll | `OMPI_MCA_coll=^hcoll,han`（非均匀分布会崩在 `coll/han`） |
| 资源口径 | 1 DCU 槽 = 8 核；CPU 基线跑在 `kshcnormal` |
| 输出 | 证据目录内 `result.txt` + 日志；文档里只写脱敏后的口径与结论 |
| 提交器 | OpenAPI 提交用 `--work-dir <workspace>/logs`，`--stdout/--stderr` 也指到 `logs/` | 提交器会在 work-dir 生成 `job_*` 记账文件，指到工作区根会污染根目录 |

标准脚本：

```bash
source $WORKSPACE/scripts/env.sh
$WORKSPACE/scripts/build-app.sh                       # 增量构建
sbatch $WORKSPACE/scripts/gate.slurm                  # trace + XB + 500 步分项 + 3x1000 步
sbatch $WORKSPACE/scripts/paired-ab.slurm             # 同二进制 paired A/B
sbatch $WORKSPACE/scripts/cpu-regression.slurm        # CPU 五算例 normal/strict
$WORKSPACE/scripts/cleanup.sh                         # 日志/快照轮转 + 占用
```

## 3. 每轮开发流程（与仓库文档一致）

1. 本地 `dev` 提交 → 推送 `origin/dev`。
2. 覆盖 `src/OneFLOW/` + 更新 `src/REVISION`，`scripts/build-app.sh` 增量编译。
3. 跑门禁（顺序固定）：3 步多区 trace + resident contract → 200 步跨后端 →
   CPU 五算例 normal/strict → 50 步稳定性；性能改动再加 paired。
4. 证据落到 `evidence/<gate>/<jobid>/`，数字写回
   `doc/plans/oneflow-development-todo.md` 的对应勾选项。
5. `scripts/cleanup.sh` 收尾；`_attic` 满一轮后清空。

## 4. 常见故障对照

| 现象 | 原因 | 处理 |
|---|---|---|
| `no ROCm-capable device is detected` | 节点未分配到 DCU / 卡被占用 | 检查 `--gres=dcu:N` 与节点，重投 |
| `The imported target "hip::galaxyhip" references "/opt/rocm/hip/lib/libgalaxyhip..." but this file does not exist` | 没设 `ROCM_PATH`，DTK 的 `hip-targets.cmake` 前缀回落 `/opt/rocm` | `export ROCM_PATH=<DTK>` 后重新 configure（见 §1 配方） |
| `Could not find a package configuration file provided by "amd_comgr"` | `enable_language(HIP)` 的 try_compile 子工程没继承目录 | 传 `-Damd_comgr_DIR/-DAMDDeviceLibs_DIR/-Dhsa-runtime64_DIR` **且** `-DCMAKE_TRY_COMPILE_PLATFORM_VARIABLES="CMAKE_PREFIX_PATH;amd_comgr_DIR;AMDDeviceLibs_DIR"` |
| `/bin/sh: .../cmake-3.30.5-linux-x86_64/bin/cmake: No such file` | `make` 用缓存里的旧 cmake 路径，且 `cmake` 未在 `PATH` | `source scripts/env.sh`（已把 `$CMAKE` 的目录前置到 `PATH`） |
| `ldd` 找不到 `libmpi.so.40` | 未 source `scripts/env.sh` | 所有作业脚本第一行 source |
| `srun` 下 MPI size=1 | `srun` 不做 MPI 启动 | 改用 `mpirun` |
| `bnd_resize` 报 10 ms/step | 首个 `hipMalloc` 被平均进短基准 | 基准先 warmup 至少一步 |
| 多 zone 卡死 / 数值发散的假象 | 每 rank 的 cell 数太小 | 固定 per-rank cell 数再比较 |
