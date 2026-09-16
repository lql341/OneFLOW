# OneFLOW 主 solver DCU vertical slice 交接

**日期：** 2026-09-16
**工作分支：** `dev`
**已推送基线：** `5ae02205`（本轮公平 benchmark 与 MPI rank-local state sync 修复；待推送）
**F1 checkpoint：** `d5005ad6`（5 个源码/测试文件，已推送）
**F1 guard：** `ca14118c`（capability/fail-fast/CMake 联动，已推送）
**F2.1 checkpoint：** `8e08c376`（主 solver adapter one-call CPU/HIP oracle，已推送）
**F2.2 checkpoint：** `156f94fc` + `2f597a04`（多帧 stage trace 与已验证模块顺序，已推送）
**F2.3 checkpoint：** `9fbbe6a7` + `fd8d9eca`（物理语义 contract 与生产 stage 守恒门禁，已推送）
**F2.4 checkpoint：** `a6ceae06`/`93955122`（3D m6 三阶段 1-step 已通过；50-step 稳定性仍失败）
**交接状态：** F2.4 3D m6 1-step/RK 与 3-step 精度门禁已通过；50-step 共同物理发散仍待收口。标准 root HIP runner 已沉淀，但当前 host-staged 单卡纵切线尚未显示端到端加速

## 1. 一句话结论

CPU vertical slice 已闭环，standalone 1D HIP contract 已在昆山真实 DCU 上
通过；当前正在做的是 **3D 主 solver 的第一条 HIP/DCU 通量纵切线**。F1 checkpoint
已经把 `UNsInvFlux` 的五方程 Lax-Friedrichs batch 数据交给可注入的
`FluxBackend`；`ca14118c` 又补齐 capability/fail-fast policy，并已在昆山用
DTK 26.04、`gfx906` 完成生产 `OneFLOW` HIP 编译链接、smoke 与 root HIP
contract。`8e08c376` 又修复 HIP Lax-Friedrichs scheme 与显式
`boundaryMask` 语义，并在真实 DCU 上完成 257 faces × 5 equations 的主 solver
adapter one-call flux/residual CPU oracle。`2f597a04` 又在 5 方程
`plateuns2dslau2` Lax-Friedrichs 小 case 上完成 legacy CPU、CPU batch、HIP batch
的 1-step LU-SGS face/residual/state trace。`fd8d9eca` 又把 connectivity/geometry
写入 opt-in stage metadata，三路都能从生产 face flux 精确重建 residual，并用独立
HIP contract 验证 boundary mask、内部面守恒、ALE 与 face-area ownership。3D m6 三阶段 1-step 已通过；50-step 在 CPU legacy 与 HIP 共用配置下均于约第 20 步出现负压/Inf/NaN，不能作为 HIP 分歧；标准 root HIP runner 已沉淀并完成 3-step benchmark，但当前 host-staged 单卡纵切线尚未显示端到端加速，因此不能宣称主 solver DCU vertical slice 已全部完成。

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
  - [ ] F2：CPU/HIP numerical gate——F2.1–F2.4 的 1-step/RK 与 3-step 门禁已完成，50-step 稳定性待解决。
  - [ ] F3：target-node evidence——构建/contract/one-call/小 case/物理语义/3D multi-step trace/CPU regression 与标准 runner 已有证据；50-step 物理稳定性、MPI/多卡与有效加速仍待完成。
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

F2.2 checkpoints `156f94fc`（多帧 stage trace）与 `2f597a04`
（昆山已验证模块顺序）继续完成：

- 新增独立 `ONEFLOW_UNS_STAGE_TRACE_FILE`，逐次追加 face、inviscid residual
  与更新后 internal-cell state；既有 E6 trace 格式保持不变；
- 新增 `ci/kunshan/f2-main-solver-stage.slurm` 与三路 trace verifier；
- 5 方程小 case 1-step LU-SGS 的 legacy CPU / CPU batch / HIP batch 已在真实
  DCU 上对齐。

F2.3 checkpoints `9fbbe6a7`（硬件 contract）与 `fd8d9eca`
（生产 stage semantics）继续完成：

- root HIP smoke 独立检查非 boundary-first 显式 mask、内部面守恒，以及 ALE
  mesh-normal velocity 与 face-area 只计算一次；
- stage trace 新增 connectivity/geometry metadata，verifier 从真实生产 face flux
  重建 `AddF2CField` residual，并检查全域 residual 与边界通量闭合；
- metadata 仍使用既有 `OFSTG01` record header；既有 E6 trace 格式不变。

接手时应先确认 `dev` 与 `origin/dev` 均包含 `fd8d9eca`；不要 checkout、reset
或覆盖 F1/F2.1/F2.2/F2.3 文件。

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
- [x] 主 solver 小 case 1-step：`plateuns2dslau2` 五方程 Lax-Friedrichs，
  legacy CPU / CPU batch / HIP batch 各生成 face、metadata、inviscid residual、state 四条记录；
  HIP 对 legacy 的 qf1/qf2 完全一致，invflux/residual 最大绝对差
  `1.4210854715202004e-14`，更新后 state 最大绝对差
  `6.514681130330882e-19`；density/pressure finite 且为正。
- [x] F2.3 小 case 物理语义：三路 production trace 的 residual 重建误差均为
  `0`，全域 residual 与边界通量闭合误差 `2.804e-13`；非 boundary-first
  synthetic HIP contract 的 boundary semantics 误差 `4.441e-16`、内部面守恒
  误差 `2.753e-14`；ALE/face-area 解析 contract 误差 `1.110e-16`。
- [x] `fd8d9eca` 根 CTest `210/210`；目标节点 root HIP GoogleTest `9/9`、
  hardware HIP CTest `10/10`，测试集合非空。
- [x] F2.2 revision `2f597a04` 昆山 CPU 门禁：根 CTest `210/210`；五算例 normal
  `5/5`（最大 residual absolute difference 约 `4.97e-10`）；strict
  `5/5`（最大约 `1.11e-17`）；standalone CPU contract `8/8`。
- [x] 相关目标节点作业均为 scheduler `COMPLETED` 且 workload exit code `0:0`。

### 4.2 尚未完成，禁止提前声明

- [x] `UNsInvFlux` HIP batch 已在真实 3D m6 case 的 3-stage 1-step 中执行。
- [x] 3D m6 三阶段 1-step 已逐 stage 对比 `qf1`、`qf2`、face flux、residual、state；HIP 对 legacy 最大差 `1.7064127888488656e-13`，finite 且 density/pressure 为正。
- [x] 3D m6 3-step 三路 trace 共 36 条记录通过；HIP 对 legacy 最大绝对差 `2.8399504969911504e-13`，最大 scaled error `1.992850329202156e-13`。
- [ ] 小 case 的 finite、positivity、boundary semantics 与守恒已通过；3D m6 1-step
  的 finite/positivity 与 trace 对齐已通过，但 50-step 在 CPU/HIP 共用配置下于约第 20 步
  出现负压/Inf/NaN，3D 长步稳定性仍未通过。
- [x] 标准 root HIP runner 已沉淀为 `ci/kunshan/f3-main-solver-benchmark.slurm`；流式 verifier 避免 3D 多步 trace 一次性读入导致 OOM。
- [x] 3D m6 3-step 三路 trace 与性能门禁：job `122246208` 为 `COMPLETED/0:0`，36 条记录全通过；CPU 对 legacy 最大绝对差 `2.8332891588433995e-13`，HIP 最大绝对差 `2.8399504969911504e-13`。
- [x] 同一 `steps=3, warmup=1, repeats=3` basis 的端到端 wall-clock：legacy `24775.415 ms`，CPU batch `25520.782 ms`（`0.970794x`），HIP/DCU batch `25224.514 ms`（`0.982196x`）；当前不能宣称主 solver DCU 加速。
- [x] 公平资源口径的 8 CPU MPI ranks vs 1 DCU：legacy `25904.228096 ms`，CPU batch `26794.339157 ms`（`0.966780x`），HIP/DCU batch `25658.172501 ms`（`1.009590x`）；HIP 仅快约 `0.95%`，尚不足以称为有意义的加速。
- [x] 为支持公平 MPI benchmark，修复 `SyncAllEulerDomainStates` 遍历全局 zone 导致非 owner rank 解引用空 `globalGrids` 的问题；改为按 `ZoneState::localZid` 同步 rank-local zones。修复后 8-rank CPU warmup、3-step trace 与 HIP contract 均通过。
- [ ] 尚未完成主 solver DCU MPI、多卡、GPU-resident/设备归约与性能优化。
- [ ] 根工程默认仍是 CPU-only；standalone HIP 通过不能替代主 solver DCU 证据。

## 4.3 3D m6 性能证据（不等于最终生产加速）

- job `122246208`：`COMPLETED/0:0`；DTK 26.04、`gfx906`、`dcu:1`，8 CPU，单 rank。
- 精度门禁：3 steps、3-stage RK，36 条 legacy/CPU batch/HIP batch trace 记录；CPU 对 legacy 最大绝对差 `2.8332891588433995e-13`，HIP 最大绝对差 `2.8399504969911504e-13`；最大 scaled error 分别为 `1.7075230118734908e-13`、`1.992850329202156e-13`。
- 性能 basis：同一输入与配置，`steps=3`、warmup `1`、repeats `3`，端到端 wall-clock 包含 `mpirun`、初始化、host pack、H2D、kernel、D2H 与输出。

| 模式 | mean lifecycle (ms) | 相对 legacy |
|---|---:|---:|
| legacy CPU | 24775.414955 | 1.000000x |
| CPU batch | 25520.781995 | 0.970794x |
| HIP/DCU batch | 25224.513536 | 0.982196x |

结论：当前正确性已成立，但该 host-staged 单 rank 单 DCU 纵切线没有端到端加速；HIP 约慢 1.78%。不能把这组结果宣传为最终 GPU 性能，下一步应优先减少 host/device 往返、复用 device buffer、合并 launch，并在修复 50-step 物理稳定性后重测。

### 4.4 公平资源口径的 8 CPU ranks vs 1 DCU

该组数据来自同一个 3D `m6wingroe_sa` 输入、同一编译 revision、同一
`steps=3,warmup=1,repeats=3` basis；legacy 与 CPU batch 使用 8 个 MPI ranks，
HIP batch 使用 1 个 DCU rank。计时仍是端到端 wall-clock，包含 MPI launch、初始化、
host pack、H2D、kernel、D2H 与输出；因此它是当前 host-staged vertical slice 的
应用级证据，不是设备 kernel-only speedup。

| 模式 | ranks/device | mean lifecycle (ms) | 相对 legacy |
|---|---:|---:|---:|
| legacy CPU | 8 MPI ranks | 25904.228096 | 1.000000x |
| CPU batch | 8 MPI ranks | 26794.339157 | 0.966780x |
| HIP/DCU batch | 1 DCU | 25658.172501 | 1.009590x |

3-step trace gate 同时通过：CPU batch 对 legacy 最大绝对差
`2.8332891588433995e-13`，HIP batch 对 legacy 最大绝对差
`2.8399504969911504e-13`；36 条记录，state finite，density/pressure 为正。
结论是当前 DCU 与 8-rank CPU 基本持平，约 `0.95%` 的优势处于运行波动范围内，
不能作为性能宣传数据。下一轮应先做 profile，拆分 MPI/初始化/pack/H2D/kernel/D2H/输出
时间，再把 state 和 buffer 尽可能留在设备侧。

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

F1 已解决 backend registration、capability 与 fail-fast policy，F2.1/F2.2 已完成
one-call 与小 case 1-step 的 CPU/HIP flux/residual/state 对比。下一步关键问题是：
- 3D m6 三阶段 1-step 已逐 stage 比较 `qf1`、`qf2`、`invflux`、residual 与 state；
- 3D 50-step 在 CPU/HIP 共用配置下共同发散，需先定位 CFL/边界/初始化/物理稳定性条件；
- 标准 root HIP runner 与 3D m6 3-step 性能门禁已完成；完整粘性 NS/HIP 路径、MPI/多卡与 GPU-resident 性能优化仍未完成；
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
- [x] F2.2：做主 solver 小 case 1-step LU-SGS gate。
  - [x] F2.2a：选择可控小 case，分别生成 legacy CPU、CPU batch、HIP batch trace。
  - [x] F2.2b：逐 face/equation 比较 `qf1`、`qf2`、`invflux`。
  - [x] F2.2c：比较 residual/state，定位误差来自 pack、kernel 还是回写。
- [x] F2.3：小 case 物理与离散语义检查。
  - [x] F2.3a：finite、positive density、positive pressure。
  - [x] F2.3b：内部面守恒和 boundary-mask/boundary ordering 语义。
  - [x] F2.3c：ALE mesh-normal velocity 与 face-area ownership 不重复计算。
- [ ] F2.4：扩大到 E6 使用的 3D m6 Lax-Friedrichs case。
  - [x] F2.4a-1：3-stage RK 1-step；896256 faces、三路 trace、HIP 最大差 `1.706e-13`。
  - [ ] F2.4a-2：50 steps；CPU/HIP 共用配置约第 20 步出现负压/Inf/NaN，需先修复稳定性条件。
  - [x] F2.4b：1-step 对照既有 CPU trace gate 与输出级 oracle；3-step 扩展门禁已在真实 DCU 完成。

### F3：昆山 target-node evidence

- [x] F3.1：使用 `kshdnormal`、`dcu:1`、`gfx906`、DTK 26.04；资源 tuple
  必须来自 `ci/kunshan/README.md`，不得自行猜测。
- [x] F3.2：标准 runner 已支持 root HIP opt-in、standalone contract 非空检查、3D trace 门禁和同 basis timing；仍不覆盖 MPI/多卡。
- [x] F3.3：记录工具链、可见设备、目标架构、scheduler completion 与 workload
  exit code；raw log 和具体账号/主机/job metadata 只留在集群 run artifacts。
- [x] F3.4：目标节点执行顺序：
  - [x] root HIP configure/build；
  - [x] HIP contract；
  - [x] 主 solver one-call；
  - [x] 小 case 1-step LU-SGS；
  - [x] 3D case：m6 3-step 三路 trace 与单卡 CPU/HIP benchmark 已在真实 DCU 节点完成。
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
- [x] 主 solver 小 case 1-step LU-SGS 与 CPU oracle 对齐；
- [x] 主 solver 3D m6 三阶段 1-step 与 CPU oracle 对齐；
- [ ] 主 solver 50-step 稳定性与 CPU oracle 对齐；
- [x] 小 case finite、positivity、conservation、boundary semantics 全部通过；
- [x] 3D m6 1-step finite/positive 与 trace 对齐；
- [ ] 3D m6 长步稳定性与完整物理语义通过；
- [ ] 3D case 全部门禁通过，再更新 living TODO；
- [x] 成功门禁的 scheduler/workload 均为 `COMPLETED`/`0:0`；50-step 失败正确传播为非零退出；
- [x] 提交文档中不含敏感或原始运行元数据。

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
