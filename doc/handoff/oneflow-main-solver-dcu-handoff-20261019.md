# OneFLOW 3D 多 zone HIP interface handoff

> 日期：2026-10-19  
> 分支：`dev`  
> 主题：3D `m6wingroe_sa` 多 zone interface endpoint topology 与 Kunshan 复核

## 当前 checkpoint

本轮以 `877d1f16` 为当前代码 checkpoint，包含此前两个 interface ownership
修正：

- `7f2c1d7f`：增加多 zone endpoint ownership contract；发送端使用
  `OwnedCell`，接收端使用 `GhostCell`。
- `5ea98a88`：提取 backend-neutral `BuildInterfaceEndpoint()`，由
  `InterfaceTask` 和 CPU/HIP 路径共享 topology mapping。
- `877d1f16`：删除误接入 `OneFLOW_GeometryCore` 的 heavyweight
  `InterfaceEndpointTest` target。该测试会把一个很小的 endpoint 检查接入大型
  geometry 静态库，造成不必要的完整本地重编译；删除不影响既有轻量
  boundary/interface contract。

工作区在本轮收口时保持干净。未做本地完整 CMake/CPU build；root solver build
和全部 3D 证据均来自 Kunshan。

## 已完成的 Kunshan 门禁

所有结果都针对 3D、单节点、8-zone 分区输入；不是 1D Euler microbenchmark。

### Build and HIP contract

- root HIP build 使用 Kunshan DTK 26.04、`gfx906`、`dcu:1`。
- GCC `16.2.0`、OpenMPI `5.0.11`、CMake `4.4.3`。
- 硬件 HIP CTest：`16/16` 通过。

### Accuracy trace

- CPU：`8 CPU cores / 8 MPI ranks`。
- HIP：`1 DCU / 8 MPI ranks`，单张 DCU 分配 8 个 rank。
- 单节点、8-zone，每个 rank 负责一个非空 zone。
- `3 steps`、fixed CFL `0.01`。
- CPU/HIP 均报告 `active_ranks=8`。
- CPU/HIP 均有 8 个 rank breakdown。
- 逐 rank stage trace 通过：`MULTIZONE_TRACE_PASS ranks=8 steps=3 cfl=0.01`。
- 当前准确路径仍是：
  `CPU gradient + CPU reconstruction + HIP flux/interface`。

### Stability

- CPU/HIP 均使用 8 MPI ranks。
- `50 steps`、fixed CFL `0.01`、3-stage RK、inviscid、limiter off。
- CPU/HIP workload exit code 均为 `0`。
- 8 个 rank 均写出 breakdown。
- 未观察到 NaN、Inf、负密度、负压力或 non-physical diagnostic。
- 结果：`MULTIZONE_STABILITY_PASS ranks=8 steps=50 cfl=0.01`。

### Paired timing

- 同一 8-zone 输入。
- `10 steps`、warmup `1`、formal repeats `5`、fixed CFL `0.01`。
- 端到端 wall-clock，包含 MPI、初始化、host packing、H2D/D2H、HIP kernel、
  interface 和输出。
- CPU：`6249.223629 ms`。
- HIP：`5868.614810 ms`。
- 比值必须写成：`CPU wall-clock / HIP wall-clock = 1.064855x`。
- 该轮 HIP 为 CPU 的 `0.939095x`，约快 `6.49%`。
- 这是当前节点和当前 revision 的复核结果，不替换正式性能报告。此前另一轮
  节点测得 CPU `3473.812429 ms`、HIP `4337.233101 ms`，两轮不能混合平均，
  也不能把单轮差异直接解释成稳定应用级加速。

## 已确认的失败边界

不能重新打开以下 opt-in：

- `ONEFLOW_ENABLE_UNS_HIP_GRADIENT`
- `ONEFLOW_ENABLE_UNS_HIP_RECONSTRUCTION`
- `ONEFLOW_ENABLE_UNS_HIP_STATE_UPDATE`
- `ONEFLOW_ENABLE_UNS_HIP_BOUNDARY`
- `ONEFLOW_ENABLE_UNS_HIP_TIME_STEP`

历史失败现象仍有效：

- 多 zone device reconstruction 曾造成 interface/boundary ghost 错配，
  `qf2.u` 最大差约 `49.3`。
- 仅打开 HIP gradient、保留 host reconstruction 时，`qf1` 最大差约
  `3.10e-3`。
- 清理残留 HIP opt-in 后，3-step trace 恢复通过，HIP 相对 CPU 最大绝对差约
  `8.53e-14`。

本轮验证中还暴露了 runner 层的两个问题，均不是 solver 数值失败：

- 通用单 zone runner 假设 case 下存在 `script/grid.txt` 的某些配置，不能直接
  用于外部 8-zone fixture。
- 50-step 自定义检查曾用过宽的字符串 grep，把普通日志中的 `inf` 子串误判为
  Inf；后续已改为 token-aware diagnostic matching。

## 下一步

按以下顺序推进，不跳过 CPU oracle：

1. 将 8-zone fixture 的准备逻辑沉淀为可复现 runner，明确 `gridObj=3`、
   `grid/m6wing-8.ofl`、boundary map、8-zone/8-rank ownership 和 runtime
   `system` deployment。
2. 增加轻量 interface/ghost contract，避免再次链接
   `OneFLOW_GeometryCore`。至少覆盖 ordinary boundary、interface、periodic、
   solid boundary、ghost gradient copy、空消息和 self-neighbor skip。
3. 明确并记录每个 rank 的 local zone、interface face、owned cell、ghost cell、
   send/receive buffer 以及 host/device memory space。
4. 先在 CPU 8-zone 上验证 ownership 和 synchronization contract，再让 HIP path
   消费同一份 endpoint map。
5. 只有 ghost synchronization contract 通过后，才尝试 HIP gradient 或
   reconstruction 的多 zone opt-in。
6. 每次修改后固定执行：
   `build -> contract -> 3-step trace -> fixed-CFL stability -> rank utilization
   -> repeated paired timing`。

## 证据边界

本 handoff 不包含原始 Slurm 日志、作业号、节点名、私有路径或凭据。原始
breakdown、stage trace、runner 输出和环境记录保留在 Kunshan workspace 的
revision-specific run artifacts 中；正式报告仍以脱敏后的 Markdown 为准。
