# OneFLOW HIP/MPI Optimization Handoff

你接手的是 OneFLOW 仓库当前 `dev` 工作树。请先完整阅读：

1. `AGENTS.md`
2. `ci/kunshan/README.md`
3. `doc/plans/oneflow-development-todo.md`
4. `doc/reports/performance/oneflow-euler-performance-current.md`
5. `doc/reports/README.md`

开工前执行并记录：

```bash
git status --short --branch
git log -5 --oneline
git branch -vv
```

本轮交接允许在保留现有用户修改的前提下完成必要的合并、提交和推送。
不要用 destructive 操作覆盖或回退无关修改；任何集群验证都必须使用
SCNet OpenAPI 的昆山后端，不能用本机编译结果替代目标节点证据。

## 当前结论

当前 3D HIP 主求解链已经完成一轮受限应用级正确性验证，并在固定 CFL 的
10-step 单次 paired timing 中显示出明显优势；但该 timing 尚未达到正式性能
报告所需的重复次数，因此仍需把稳定性、MPI correctness、rank utilization 和
多次同口径 paired timing 作为后续门禁。

最新正式 3-step trace 配置为单节点、单 rank、单 DCU、五方程、
`ischeme=5` Lax-Friedrichs、inviscid、limiter off、3-stage RK、
`warmup=1`、`repeats=3`。结果：

- root HIP smoke 成功，GoogleTest `15/15` 通过，HIP CTest `16/16` 通过。
- `STAGE_TRACE_PASS records=36`；HIP state、physical state、conservation 和
  face metadata 均通过，最大 state 误差约 `2e-15`。
- 同一 basis 的 3-step timing 为：legacy `5870.210048/5830.465024/5827.696896 ms`，
  CPU batch `6811.954688/6856.023040/6832.440064 ms`，HIP batch
  `2761.182976/2767.150592/2743.480832 ms`。短算例明显受初始化和启动固定成本影响，
  因此这组数据用于诊断，不单独作为正式应用加速声明。

补充的 10-step fixed-CFL paired timing 使用同一输入、单 rank、单 DCU、
`CFL=0.001`、无 trace、每个模式一次运行：

- legacy：`16417.506 ms`
- CPU batch：`16458.731 ms`
- HIP batch：`3870.999 ms`

HIP 相对 legacy 约 `4.24x`。这是当前最直接的“完整受限应用路径”性能信号，
但只有单次运行，不能替代正式多次 paired timing，也不能外推到跨节点或完整
湍流求解器。

20-step、`CFL=0.01`、inviscid、limiter off 配置曾在 CPU 第 13 步出现负压力，
因此该配置不能作为稳定长跑性能依据；后续性能必须使用明确的 fixed-CFL 稳定配置，
并同时记录 workload exit code 和物理诊断结果。

当前可报告的结论必须分层：

1. HIP kernel/阶段加速：已通过受限路径的 trace 和 device kernel 正确性验证。
2. 完整应用加速：10-step fixed-CFL 单次结果约 `4.24x`，尚未完成重复 timing
   门禁。
3. 跨节点扩展：尚未形成可公开的 paired timing 证据，不能宣称。

已通过的边界：

- root HIP smoke、GoogleTest `15/15` 和 HIP CTest `16/16` 通过。
- 合法五方程受限 HIP：3-step、36 条 stage trace 通过，state 最大误差约 `2e-15`。
- GPU-aware MPI：设备指针 probe 仅在已知可用配置下通过；不能外推到所有 MPI、
  UCX 或集群配置。
- 湍流算例：`test/cfd/turbplateuns2droe_sa` 的 HIP 平均流 + CPU 湍流混合路径通过；不能宣称完整湍流 HIP。
- 非法三方程配置现在 fail-fast。当前 compressible NS 状态契约仍是五分量 `(rho,u,v,w,p)`；二维仍保存退化的 `w` 分量，不要把 `nEqu=3` 当作已实现的二维 NS。
- 缺失 wall-distance 文件已有自动计算/策略隔离；不要把 fixture 缺失文件当作 GPU 算法错误。

## 2026-10-19 rank/DCU 矩阵与 zone 分区边界

在 `af7580cc` 上，Kunshan 单节点 `kshdnormal` 完成 standalone 1D Euler
rank/DCU 矩阵：CPU 使用 8/16/32 MPI ranks（每 rank 1 core），HIP 使用
1/2/4 MPI ranks（每 rank 1 DCU），全局规模为 65,536、262,144、1,048,576
和 4,194,304，固定 `steps=100,warmup=1,repeats=2`。36 个 CPU/HIP 配对
全部通过。每组均满足 `local_ranks == requested ranks`、
`decomposition_active_ranks == requested ranks`、非空 local partition、预期
halo 次数、HIP `visible_devices == DCU ranks`、CPU/HIP final hash 一致和
`decomposition=PASS`。完整表格见
[`oneflow-1d-euler-mpi-rank-dcu-matrix-20261019.md`](../reports/performance/oneflow-1d-euler-mpi-rank-dcu-matrix-20261019.md)。

这里没有“按 zone 重划”的步骤，因为该 standalone 端口没有 zone；代码在每次
`MPI_Comm_size` 后按 `Partition(globalNx, rank, ranks, ...)` 重新切连续 cell
区间，halo 邻居也随 rank ring 改变。相反，3D `m6wingroe_sa` 是 single-zone，
HIP capability 要求 `ZoneState::nLocal == 1`，不能把单 zone 通过增加 MPI rank
包装成有效 8/16/32-rank 或 1/2/4-DCU zone 实验。3D 多 rank 下一步必须准备
已分区 multi-zone fixture，重新生成 `ownerRanks` 和 interface exchange plan，
并验证每个 rank 的 local zone 非空且实际写出 solver work。

## 主要优化目标

### 1. HIP 整体路径

将当前“HIP 平均流 + CPU 旁路”逐步推进到真正的设备驻留路径：

```text
device state
→ gradient
→ face reconstruction
→ inviscid flux
→ residual accumulation
→ Runge-Kutta state update
→ boundary update
→ halo/interface exchange
→ ghost-cell update
```

保持 CPU oracle。每个阶段都要有 CPU/HIP 数值比较和失败门禁。

重点检查：

- `codes/uns/src/UNsInvFlux.cpp`
- `codes/uns/src/UNsUpdate.cpp`
- `codes/ns/src/NsSolverImp.cpp`
- `codes/accel/src/HipFluxBackend.hip`
- `codes/accel/include/HipFluxBackend.h`
- `codes/parallel/include/BoundaryExchangeContract.h`
- `codes/parallel/include/DistributedDomainPlan.h`
- `codes/task/src/InterfaceTask.cpp`

当前已知的性能风险：

- `UpdatePrimitiveState(view, true)` 会把设备状态拷回主机。
- 状态合法性检查有 device-to-host 往返。
- 每个阶段可能重复分配、上传、下载和同步。
- 多 zone 分区边界数据交换仍可能经过 host staging。
- HIP path 目前不是所有湍流方程、源项、粘性项和 `turbq/turbres` 的完整 device state。
- 小规模四 zone 算例中，初始化、MPI 启动、文件输出和通信可能淹没 kernel 计算收益。

优化原则：

- 优先持久化 device state，避免每个阶段重复分配。
- 将 host/device copy 变成显式、可计时、可验证的契约。
- 尽量只在 CPU oracle、输出或确有必要的边界点下载。
- 不能为了 benchmark 临时关闭正确性检查。
- 不新增 process-wide persistent GPU state。
- 设备所有权必须唯一，不能让 CPU 和 HIP 同时隐式拥有同一字段。

### 2. MPI 加速路径必须 backend-neutral

之前的 MPI 多 zone 修复和 GPU-aware MPI 不能写成 HIP 专用逻辑。所有 backend 都必须复用同一套：

- `DistributedDomainPlan`
- 分区边界数据交换计划（halo exchange plan）
- rank/zone ownership
- 确定性的 zone pair 顺序
- 自身邻居跳过规则
- MPI 调度和阻塞通信顺序
- 空消息处理
- correctness、rank-utilization 和 paired timing 门禁

backend 只实现自己的部分：

- device state
- `PackBoundary`
- `UnpackBoundary`
- host/device copy
- numerical kernels
- device capability

MPI 层不得通过 HIP 类型判断是否交换，也不得把 CUDA/HIP API 写进通用通信调度。

必须保留以下扩展位：

```text
CPU host buffer
HIP device buffer
未来 CUDA/Kokkos/其他 backend buffer
```

建议把传输能力抽象为：

```text
BoundaryMemorySpace
BoundaryTransferMode
BoundaryExchangeContract
DistributedDomainPlan
Backend capability query
```

第一版默认使用通用 host-staged MPI：

```text
device boundary
→ backend PackBoundary
→ Device-to-Host
→ MPI exchange
→ Host-to-Device
→ backend UnpackBoundary
→ ghost cells
```

只有在设备指针 probe、单节点 correctness、双节点 correctness 和重复 timing 全通过后，才允许使用 GPU-aware MPI 的 device-direct 路径。GPU-aware MPI 是 MPI 直接接受设备内存指针的能力，不代表所有 MPI/UCX 配置都支持。

### 3. 多 zone 和自定义邻居

继续保证：

```text
CUpdateInterface::Run()
→ SwapInterfaceData()
→ sendAction()
→ HXSwapData()
→ recvAction()
```

要求：

- zone pair 是确定顺序的无向集合。
- 每个 pair 连续执行两个方向。
- 不同 MPI rank 不得以不同顺序进入阻塞通信。
- `z2n` 必须映射到真实 pair 索引，不能复用邻居循环下标。
- 自身邻居必须明确跳过，除非数据结构契约明确要求 self-copy；默认不要对自身执行 MPI。
- 零长度接口消息必须安全。
- 自定义邻居、稀疏接口和多 zone 必须覆盖 contract tests。

## 必须新增或完善的验证

验证顺序固定为：

```text
build
→ contract tests
→ CPU oracle
→ 1/3-step multi-zone accuracy
→ fixed-CFL stability
→ MPI correctness
→ rank utilization
→ 多次同口径 paired timing
```

每次性能结论必须同时给出：

- 后端和具体 feature flags
- 方程数、通量格式、CFL、时间积分和 RK stage
- zone 数、MPI rank 数、节点数、DCU 数
- warmup 和 formal repeats
- wall time 是否包含启动、网格、通信、计算和输出
- CPU/HIP 数值比较结果
- workload exit code 和 scheduler 状态

禁止比较不同 repeats、不同步数、不同 CFL、不同输出规则或不同 backend feature flags 的数据。

CTest 最终必须使用非空测试门禁；空测试集、`discover_tests undefined`、`No tests were found` 和 `_NOT_BUILT` 都不能算通过。

## 湍流、跨节点和后续 backend

当前湍流路径是：

```text
HIP 平均流
+
CPU 湍流方程、源项、粘性项和 turbq/turbres 通信
```

后续接入完整湍流 HIP 时，必须为湍流状态建立 backend-neutral ownership 和边界交换契约，不能只给平均流增加 device buffer。

跨节点 GPU-aware MPI 只有在：

- 双节点设备指针 probe
- 双节点四 zone correctness
- host-staged 与 device-direct 数值一致
- 多次 paired timing

全部通过后才能外推到跨节点性能。UCX endpoint 错误要记录为具体配置限制，不要默认为 OneFLOW 算法错误。

## 代码和报告边界

- 不更新 `doc/plans/oneflow-development-todo.md`、正式 handoff 或正式性能报告，除非用户明确要求。
- 不把集群日志、绝对路径、节点名、账号、job metadata 写入公开报告。
- 不宣称完整湍流 HIP、所有 multi-zone 完成、所有 MPI 配置支持 GPU-aware MPI 或自动 METIS 分区完成。
- 正确性、稳定性、MPI correctness 和重复 timing 全通过前，不宣称加速。
- 所有新专业缩写首次出现时解释：分区边界数据交换（halo exchange）、幽灵单元（ghost cell）、GPU-aware MPI。

## 建议执行顺序

1. 审查 `HipFluxBackend` 的 state lifecycle、同步点、重复分配和 D2H 路径。
2. 增加阶段级 breakdown：初始化、gradient、reconstruction、flux、residual、state update、boundary、MPI interface、D2H/H2D、output。
3. 在单 zone CPU oracle 上先启用完整 HIP gradient/reconstruction/state update/boundary。
4. 逐阶段加入多 zone 和 halo exchange。
5. 默认 host-staged MPI；设备直连 MPI 只作为 capability-selected fast path。
6. 对四 zone 做 1/3-step accuracy 和 fixed-CFL stability。
7. 再做单节点 1/2/4 rank 和双节点 paired timing。
8. 只有当完整应用 wall time 小于 CPU 且数值门禁通过，才报告 HIP speedup。
9. 最后接入完整湍流 HIP、跨节点扩展和其他 backend。

最终汇报必须明确区分：

```text
正确性通过
性能接近
阶段级 kernel 加速
完整应用加速
跨节点扩展
```

这些不是同一个结论，不能混写。
