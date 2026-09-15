# OneFLOW 主 solver DCU vertical slice 交接

**日期：** 2026-09-15
**工作分支：** `dev`
**已推送基线：** `8e08c376`（`origin/dev`）
**F1 checkpoint：** `d5005ad6`（5 个源码/测试文件，已推送）
**F1 guard：** `ca14118c`（capability/fail-fast/CMake 联动，已推送）
**F2.1 checkpoint：** `8e08c376`（主 solver adapter one-call CPU/HIP oracle，已推送）
**交接状态：** F2.1 one-call 已通过；下一步是 F2.2 one-step/one-stage

## 1. 一句话结论

CPU vertical slice 已闭环，standalone 1D HIP contract 已在昆山真实 DCU 上
通过；当前正在做的是 **3D 主 solver 的第一条 HIP/DCU 通量纵切线**。F1 checkpoint
已经把 `UNsInvFlux` 的五方程 Lax-Friedrichs batch 数据交给可注入的
`FluxBackend`；`ca14118c` 又补齐 capability/fail-fast policy，并已在昆山用
DTK 26.04、`gfx906` 完成生产 `OneFLOW` HIP 编译链接、smoke 与 root HIP
contract。`8e08c376` 又修复 HIP Lax-Friedrichs scheme 与显式
`boundaryMask` 语义，并在真实 DCU 上完成 257 faces × 5 equations 的主 solver
adapter one-call flux/residual CPU oracle。one-step/one-stage 和 3D case 数值
门禁仍未完成，因此不能宣称主 solver 已具备 DCU 能力。

## 2. 全景进度

- [x] 阶段 A：运行基线——同步 fork/upstream，固定可复现的共同开发基线。
- [x] 阶段 B：standalone——用 1D Euler 建立 CPU/HIP lifecycle、trace、MPI
  和真实 DCU 最小验证闭环。
- [x] 阶段 C：accelerator substrate——完成 `AccelRuntime`、
  `AccelBackend`、`FluxBackend` 与 `DeviceBuffer` 基础设施。
- [x] 阶段 D：domain contract——明确 solver/backend 的数据布局、几何、
  capability、state owner 和初始化/restart 生命周期。
- [x] 阶段 E：CPU vertical slice——3D 主 solver 已完成 CPU batch seam、
  INIT/restart、RK 调度、legacy oracle、逐面 trace 与物理不变量验收。
- [ ] 阶段 F：DCU vertical slice——把同一主 solver batch contract 切到 HIP，
  在真实 DCU 上完成编译、运行和 CPU/HIP 数值一致性。
  - [x] F1：HIP backend registration——`d5005ad6` + `ca14118c`，目标节点编译与 contract 已通过。
  - [ ] F2：CPU/HIP numerical gate——F2.1 one-call 已完成，当前进入 F2.2 one-step/one-stage。
  - [ ] F3：target-node evidence——构建/contract/one-call/CPU regression 已有证据；主 solver case 待运行。
- [ ] 阶段 G：MPI/性能——在单卡正确性闭环后再做 halo、多卡和性能优化。

## 3. Git 状态与 F1 checkpoint

F1 源码/测试修改已保存为 checkpoint commit `d5005ad6`（`wip: connect main solver batch flux to HIP backend`）。该 checkpoint 的父提交 `85a897a0` 是交接更新前的 `origin/dev` 基线，已经包含：

- upstream/master `a6c81105` 的合并提交 `6d30b783`；
- 根工程 CPU+MPI 完整构建和 CTest `206/206` 通过的基线；
- F 阶段分解与统一 HIP CTest 注册设计。

checkpoint `d5005ad6` 只修改以下 5 个文件，共 147 行新增、2 行删除：

- `codes/accel/include/EulerCpuAdapter.h`
- `codes/accel/src/EulerCpuAdapter.cpp`
- `codes/uns/include/UNsInvFlux.h`
- `codes/uns/src/UNsInvFlux.cpp`
- `tests/euler_cpu_adapter_test.cpp`

这些修改不是垃圾文件，不要 checkout、reset 或覆盖。其意图是：

1. 给 `EulerCpuAdapter::CalcInvFlux` 增加 `FluxBackend&` 注入重载；
2. primitive-to-conserved 与 equation-major pack 保持 CPU/HIP 共用；
3. `UNsInvFlux` 增加 `ONEFLOW_ENABLE_UNS_HIP_BATCH=1` opt-in 路径；
4. CPU 与 HIP 都进入统一的 `CalcInvFluxBatch(FluxBackend&)`；
5. 增加 recording backend contract test，证明 adapter 只提交一次完整 packed
   contract，并保留 scheme 参数。

后续提交 `ca14118c`（`feat: guard main solver HIP flux path`）又完成：

- `ONEFLOW_ENABLE_HIP_TESTS=ON` 联动生产 `ONEFLOW_ENABLE_HIP=ON`；
- 抽取纯数据 `EulerInvFluxCapability`，逐项检查 build/runtime/backend、
  solver、local zone、grid level/count、方程数、limiter 和通量格式；
- 显式请求 HIP 但条件不满足时按稳定 reason fail-fast，不静默回退；
- capability accept/reject/first-blocking-reason contract test。

F2.1 checkpoint `8e08c376`（`feat: validate main solver HIP flux one-call`）
继续完成：

- 修复 `HipFluxBackend::CalcInvFlux` 忽略 `scheme` 的 contract 漂移，使
  `scheme=1` 与 CPU Roe-平均 Lax-Friedrichs oracle 一致；
- residual kernel 消费显式 `boundaryMask`，不再假定 boundary-first ordering；
- 新增 257 faces × 5 equations 的 adapter one-call，覆盖 3D normals、ALE
  mesh-normal velocity、face area 与非 boundary-first mask；
- 注册 `HIP.MainSolverFluxOneCall`，并标记 `hardware;hip;dcu`。

接手时应先确认 `dev` 与 `origin/dev` 均包含 `8e08c376`；不要 checkout、reset
或覆盖 F1/F2.1 文件。

## 4. 已完成验证与能力边界

### 4.1 已完成

- [x] merge 后根工程 CPU+MPI 100% 编译并链接 `OneFLOW`。
- [x] merge 后根工程 CTest：`206/206` 通过。
- [x] F1 checkpoint 下重新编译 `oneflow_euler_cpu_adapter_test` 与 `OneFLOW`：通过。
- [x] F1 checkpoint 聚焦测试：`EulerCpuAdapter|EulerBackendContract`，`16/16`
  通过。
- [x] `git diff --check`：通过。
- [x] standalone 1D HIP contract：昆山 DTK 26.04、`gfx906`、`dcu:1`，
  GoogleTest/CTest `9/9` 通过。
- [x] 根工程 HIP opt-in：昆山 DTK 26.04、`gfx906`、`dcu:1`，生产
  `OneFLOW` 与 `OneFLOWHipSmoke` 编译链接通过。
- [x] root HIP smoke：设备识别、double self-test、scalar flux 与 Euler flux
  全部通过。
- [x] root HIP contract：GoogleTest `9/9`；CTest `hardware` label + `HIP`
  筛选 `10/10`，包含 `HIP.MainSolverFluxOneCall`，测试集合非空。
- [x] 主 solver adapter one-call：257 faces × 5 equations；CPU/HIP 全量 flux
  最大绝对差 `6.661e-16`，显式 mask residual 最大绝对差 `1.110e-15`；
  覆盖 3D normals、ALE mesh-normal velocity、face area 与非 boundary-first mask。
- [x] 同 revision 昆山 CPU 门禁：根 CTest `210/210`；五算例 normal
  `5/5`（最大 residual absolute difference 约 `4.97e-10`）；strict
  `5/5`（最大约 `1.11e-17`）；standalone CPU contract `8/8`。
- [x] 两个目标节点作业均为 scheduler `COMPLETED` 且 workload exit code `0:0`。

### 4.2 尚未完成，禁止提前声明

- [ ] `UNsInvFlux` HIP batch 尚未在真实 3D 主 solver case 中执行。
- [ ] 尚无主 solver one-step/one-stage 的 `qf1`、`qf2`、face flux、residual
  或 state trace 对比；当前证据只到 adapter one-call。
- [ ] 尚未证明主 solver HIP 路径的 density/pressure positivity、finite、边界语义
  与守恒。
- [ ] 尚未完成主 solver DCU MPI、多卡或性能测试。
- [ ] 根工程默认仍是 CPU-only；standalone HIP 通过不能替代主 solver DCU 证据。

## 5. 架构判断

当前 seam 的方向合理：

```text
UNs reconstructed primitive faces
  -> EulerCpuAdapter（primitive -> conserved + equation-major pack）
  -> FluxBackend（CpuFluxBackend 或 HipFluxBackend）
  -> equation-major face flux
  -> 主 solver 既有 invflux / residual 路径
```

`HipFluxBackend` 当前是无成员的轻量执行对象，内部每次调用通过 `DeviceBuffer`
使用 `AccelRuntime` 分配和复制设备内存。因此在 `UNsInvFlux` 中栈上构造 backend
本身不是 state 生命周期错误。`SimuContext::AccelStates()` 管理的是跨阶段复用的
`EulerDomainState`，不要为了形式统一把无状态 `HipFluxBackend` 强行塞进 registry。

F1 已解决 backend registration、capability 与 fail-fast policy，F2.1 已完成
同一 5 方程 face batch 的 CPU/HIP one-call 全量 flux/residual 对比。下一步关键问题是：
- 在主 solver one-step/one-stage 中比较 `qf1`、`qf2`、`invflux`、residual 与 state；
- 检查 finite、density/pressure positivity、内部面守恒与 boundary semantics；
- 当前 host pack + H2D + kernel + D2H 只用于正确性纵切线，不代表最终性能架构。

## 6. 详细 TODO（严格按顺序）

### F0：接手与工作树保护

- [x] F0.1：分别执行 `git fetch origin --prune` 与 `git fetch upstream --prune`，只检查，不覆盖 checkpoint。
- [x] F0.2：确认 `dev` 基线至少包含 `85a897a0`，且包含 F1 checkpoint `d5005ad6`。
- [x] F0.3：确认 §3 的 5 文件修改只存在于 checkpoint `d5005ad6`；checkpoint 创建前已运行 `git diff --check`。
- [x] F0.4：checkpoint `d5005ad6` 已推送到 `origin/dev`；本文档以独立提交发布，保证跨机器可恢复。
- [x] F0.5：已阅读 `AGENTS.md`、`ci/kunshan/README.md`、
  `doc/plans/oneflow-development-todo.md` 和本文档。

### F1：主 solver HIP backend registration

- [x] F1.1：完成统一 backend seam。
  - [x] F1.1a：adapter 支持注入 `FluxBackend&`（checkpoint `d5005ad6`）。
  - [x] F1.1b：CPU/HIP 共用 primitive pack 与 batch 调用（checkpoint `d5005ad6`）。
  - [x] F1.1c：recording backend contract test（checkpoint `d5005ad6`，已在 CPU 通过）。
  - [x] F1.1d：审计 CMake：HIP 源只在 opt-in 条件下参与根工程编译，普通 CPU
    build 不依赖 DTK/HIP headers 或 runtime。
  - [x] F1.1e：在 DTK 26.04 工具链上编译链接生产 `OneFLOW`、
    `OneFLOWHipSmoke` 和 root HIP contract target。
- [x] F1.2：统一 capability contract。
  - [x] F1.2a：显式要求 5 方程 + Lax-Friedrichs + limiter 5 方程。
  - [x] F1.2b：显式要求 accelerator runtime 已初始化且为 HIP/DCU。
  - [x] F1.2c：抽取独立 `EulerInvFluxCapability`，加入 solver、单 local zone、
    finest/single grid 等通量 seam 条件；RK 整段约束仍由 E5 guard 独立负责。
  - [x] F1.2d：增加 capability accept/reject/first-blocking-reason 单元测试，
    每个拒绝原因可观测。
- [x] F1.3：定义 fallback/error policy。
  - [x] F1.3a：默认环境变量关闭时保持 legacy 路径不变。
  - [x] F1.3b：CPU batch opt-in 保持既有行为。
  - [x] F1.3c：HIP 被显式请求但 build/runtime/backend/capability 不满足时
    fail-fast；不静默回退。
  - [x] F1.3d：为无 HIP build、未初始化 runtime、错误 backend、scheme、
    方程数及 zone/grid 条件增加 contract test。

### F2：CPU/HIP numerical gate

- [x] F2.1：先做小型 one-call contract。
  - [x] F2.1a：同一 3/5 方程 face batch 分别调用 CPU/HIP backend。
  - [x] F2.1b：比较所有 equation/face flux，使用既有 CPU oracle 容差。
  - [x] F2.1c：确保 CTest 名称有 `CPU.`/`HIP.` 前缀及 `hardware;hip;dcu` label。
- [ ] F2.2：做主 solver one-step/one-stage gate。
  - [ ] F2.2a：选择可控小 case，分别生成 legacy CPU、CPU batch、HIP batch trace。
  - [ ] F2.2b：逐 face/equation 比较 `qf1`、`qf2`、`invflux`。
  - [ ] F2.2c：比较 residual/state，定位误差来自 pack、kernel 还是回写。
- [ ] F2.3：物理与离散语义检查。
  - [ ] F2.3a：finite、positive density、positive pressure。
  - [ ] F2.3b：内部面守恒和 boundary-mask/boundary ordering 语义。
  - [ ] F2.3c：ALE mesh-normal velocity 与 face-area ownership 不重复计算。
- [ ] F2.4：扩大到 E6 使用的 3D m6 Lax-Friedrichs case。
  - [ ] F2.4a：先 1 step，再 50 steps；不要直接跑长作业掩盖早期差异。
  - [ ] F2.4b：对照既有 CPU trace gate 与输出级 oracle。

### F3：昆山 target-node evidence

- [x] F3.1：使用 `kshdnormal`、`dcu:1`、`gfx906`、DTK 26.04；资源 tuple
  必须来自 `ci/kunshan/README.md`，不得自行猜测。
- [ ] F3.2：更新标准 runner，使 root HIP opt-in 与 standalone contract 共用
  CTest 注册 helper 和非空测试检查。
- [x] F3.3：记录工具链、可见设备、目标架构、scheduler completion 与 workload
  exit code；raw log 和具体账号/主机/job metadata 只留在集群 run artifacts。
- [ ] F3.4：目标节点执行顺序：
  - [x] root HIP configure/build；
  - [x] HIP contract；
  - [x] 主 solver one-call；
  - [ ] one-step/one-stage；
  - [ ] 3D case；
  - [x] 同 revision 的 CPU regression。
- [x] F3.5：CPU 五算例 normal `1e-8`、strict `1e-15` 与相关根 CTest 全通过后，
  才能把 F 标记完成。

### G：F 完成后的后续工作

- [ ] G1：host-staged halo correctness。
- [ ] G2：GPU-aware MPI capability probe，不假定环境支持。
- [ ] G3：多 rank/device mapping、reduction 与错误传播。
- [ ] G4：设备端 finite/positivity/error reduction，只回传标量。
- [ ] G5：四规模 correctness 与性能复测；所有比较保持相同 `repeats` 口径。
- [ ] G6：只有单卡正确性、CPU 回归、MPI 正确性分别有证据后，才报告端到端性能。

## 7. 推荐复现命令

本地 CPU 快速验证应使用 out-of-tree build，并加载本机已有的 CMake、OpenMPI、
METIS 和 CGNS modules。OpenMPI 4.x 配置需要：

```bash
-DCMAKE_CXX_FLAGS=-DOMPI_SKIP_MPICXX
```

接手者应从现有 cache 或 `ci/kunshan/README.md` 取得依赖变量，不要把个人绝对路径
写入提交。最小验证顺序：

```bash
cmake --build <root-cpu-build> --target oneflow_euler_cpu_adapter_test OneFLOW --parallel 8
ctest --test-dir <root-cpu-build> -R 'EulerCpuAdapter|EulerBackendContract' --output-on-failure
ctest --test-dir <root-cpu-build> --output-on-failure
git diff --check
```

HIP 验证只在目标 DCU compute node 上计为证据：

```bash
cmake -S . -B <root-hip-build> \
  -DONEFLOW_ENABLE_HIP_TESTS=ON \
  -DCMAKE_HIP_COMPILER=<dtk-hip-compiler> \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  <documented MPI/METIS/CGNS arguments>
cmake --build <root-hip-build> --parallel 8
ctest --test-dir <root-hip-build> -L hardware -R HIP --output-on-failure
```

以上 root HIP 配置入口已在昆山 DTK 26.04/`gfx906` 环境验证；依赖路径仍必须来自目标
集群的私有配置，不能照搬到其他集群。

## 8. 常见陷阱

- 不要把 `No tests were found` 的退出码 0 当成 HIP 测试成功。
- 不要用 login node 的 module/编译结果代替 compute-node 证据。
- 不要因 `HipFluxBackend` 栈上构造就误判 state owner；先区分无状态执行对象与
  `EulerDomainState` 生命周期缓存。
- 不要静默 fallback：显式请求 HIP 时，设备不可用应让验证失败。
- 不要把 host pack/H2D/D2H 正确性纵切线宣传成最终性能实现。
- 不要混用不同 `repeats` 的 `lifecycle_*_ms` 数据。
- 不要把 raw Slurm/CI log、绝对私有路径、hostname、账号、job id 写入仓库。
- 数值/backend 改动后必须跑 CPU 五算例 normal+strict；HIP 改动还必须跑 HIP
  contract 和目标节点 workload。

## 9. 收尾 Definition of Done

F 阶段只有同时满足以下条件才可勾选完成：

- [ ] 代码按逻辑拆分提交，`origin/dev` 可恢复且工作树干净；
- [x] 普通 CPU build 不依赖 HIP，完整根 CTest 通过；
- [x] CPU 五算例 normal/strict 通过；
- [x] 根工程 HIP opt-in 在昆山真实 DCU 节点编译、链接并发现预期测试；
- [x] HIP contract 全部通过且测试集合非空；
- [ ] 主 solver one-step/one-stage 与 CPU oracle 对齐；
- [ ] finite、positivity、conservation、boundary semantics 全部通过；
- [ ] 3D case 通过，再更新 living TODO；
- [x] scheduler 状态和 workload exit code 均成功；
- [ ] 提交文档中不含敏感或原始运行元数据。

## 10. 首要阅读文件

- `AGENTS.md`
- `doc/plans/oneflow-development-todo.md`
- `ci/kunshan/README.md`
- `codes/uns/src/UNsInvFlux.cpp`
- `codes/accel/include/EulerCpuAdapter.h`
- `codes/accel/src/EulerCpuAdapter.cpp`
- `codes/accel/include/HipFluxBackend.h`
- `codes/accel/src/HipFluxBackend.hip`
- `tests/euler_cpu_adapter_test.cpp`
- `cmake/OneFLOWEulerContract.cmake`
