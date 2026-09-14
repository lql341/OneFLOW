# OneFLOW 1D Euler DCU 移植阶段清单

更新时间：2026-09-04  
状态依据：当前仓库源码审计、当前 CPU-only CTest，以及 handoff 中已记录的昆山证据

复选框只表示对应交付物是否达到门槛；“standalone 已完成”不等于“主 OneFLOW solver 已完成”。

## 阶段 0：OneFLOW 整体架构审计

- [x] 梳理主程序、`FieldSimu`、`SolverMap`、`Multigrid`、`TimeIntegral` 调用链
- [x] 梳理 `AccelRuntime`、`AccelBackend`、`FluxBackend` 的职责边界
- [x] 梳理 `MRField`、zone、grid level、solver index 的状态所有权
- [x] 梳理 MPI/interface、restart、residual、checkpoint/output 边界
- [x] 对照 standalone 1D Euler，确认主 solver 接入缺口
- [x] 形成 CPU/HIP/CUDA/KOKKOS 能力边界表
- [x] 明确 upstream、fork DCU 分支和本地 artifact 的代码分流

交付物：[架构审计报告](oneflow-1d-euler-dcu-architecture-audit-20260904.md)

## 阶段 1：CPU baseline 和公共回归骨架

- [x] standalone CPU `CreateState/Upload/Advance/Download` contract
- [x] standalone CPU `FullTrace`、`NoTrace`、state reuse 和非法请求测试
- [x] 当前仓库 CPU-only 配置成功
- [x] 当前仓库 Euler contract CTest：5/5 通过
- [x] 当前仓库完整 CTest：7/7 通过；无 `NOT_BUILT` 测试
- [x] 保留 CPU 作为 device numerical reference 的执行路径
- [ ] 将同一 Euler case 接入 OneFLOW 主 solver runner
- [ ] 主 solver runner 覆盖 final state、residual、finite、positive state、boundary 和 conservation
- [ ] 主 solver runner 覆盖 checkpoint/restart 读写闭环
- [ ] 建立不依赖同一实现自比的独立 CPU oracle 或固定 reference

当前判定：standalone/contract 子阶段完成；主 solver CPU vertical slice 未完成。

## 阶段 2：backend-neutral 1D Euler vertical slice

- [x] standalone lifecycle 语义固定为 `CreateState → Upload → Advance → Download`
- [ ] 从 standalone contract 提取主工程 domain contract
- [ ] 建立不暴露 `MRField` 内部实现的 `ProblemView/FieldView`
- [ ] 建立 solver-aware backend factory/capability 判断
- [ ] state manager 按 `solverIndex + localZoneId + gridLevel + backend` 管理
- [ ] 在 `INIT_FLOWFIELD` 完成后创建并 upload 主 solver state
- [ ] 在 `TimeIntegral::RungeKutta` 建立 solver-aware fast path
- [ ] 主 solver CPU backend 与现有 CPU task 结果一致
- [ ] 主 solver restart、residual、convergence 和输出语义一致

当前判定：设计完成，代码接入未开始。

## 阶段 3：DCU device execution 和数据驻留

- [x] standalone HIP/DCU Face、ResidualUpdate 和 state update kernel
- [x] standalone HIP persistent device buffers
- [x] standalone `NoTrace` 单次 H2D/最终 D2H 路径
- [x] standalone `FullTrace` correctness 路径
- [x] 昆山 standalone 单卡数值与生命周期证据已记录
- [ ] 主 OneFLOW solver device state 创建、上传和销毁
- [ ] 主 OneFLOW solver 多步 NoTrace device execution
- [ ] 主 OneFLOW solver FullTrace/CPU oracle 对照
- [ ] device-side residual reduction 和 physical checks
- [ ] 主 solver 统计 allocation、kernel launch、sync、H2D、D2H

当前判定：standalone DCU execution 完成；主 solver DCU execution 未完成。

## 阶段 4：MPI、halo 和多卡

- [x] standalone CPU MPI halo exchange
- [x] standalone HIP MPI host staging：device pack → D2H → MPI → H2D
- [x] handoff 中已有单节点 4-rank/4-DCU HIP MPI 证据
- [ ] 主 solver 抽象 `PackHalo/ExchangeHalo/UnpackHalo`
- [ ] 主 solver host-staged MPI fallback
- [ ] 主 solver rank/device mapping 检查
- [ ] 目标节点 GPU-aware MPI probe
- [ ] device-buffer direct MPI
- [ ] 通信/计算 overlap 与 4 卡扩展性能

当前判定：standalone MPI 已有实验路径；主 solver halo/MPI 未接入；GPU-aware MPI 未验证。

## 阶段 5：端到端性能工程

- [x] standalone trace 型 Euler benchmark
- [x] standalone stateful NoTrace benchmark
- [x] standalone 记录 CPU/HIP wall-clock、kernel、同步和数据搬运
- [x] handoff 中已有四种规模的单卡 DCU stateful 结果
- [ ] 主 solver 完整 wall-clock：初始化 + H2D + compute + sync + MPI + reduction + output
- [ ] 主 solver CPU 单核/多核与单卡 DCU 对照
- [ ] 主 solver 统计 allocation、kernel launch、sync、H2D、D2H 和 MPI
- [ ] 每项优化重新通过 numerical/physical/restart 门禁
- [ ] 性能结论与 standalone/主 solver 范围严格分开

当前判定：standalone 性能基线完成；主 solver 端到端性能未开始。

## 阶段 6：其他 backend 兼容性

- [x] CPU backend 作为默认执行实现
- [x] HIP/DCU backend 作为当前主要移植目标
- [ ] HIP/DCU 主 solver integration evidence
- [ ] CUDA backend 独立构建、runtime、numerical、lifecycle、MPI evidence
- [ ] KOKKOS backend 独立构建、runtime、numerical、lifecycle、MPI evidence
- [ ] accelerator selection 按 backend kind/capability，而非仅 `IsAccelerator()`
- [ ] CPU/HIP/CUDA/KOKKOS 共享 solver-domain contract

当前判定：CPU 完成；HIP standalone 完成但主 solver 未完成；CUDA/KOKKOS 未验证。

## 阶段 7：经验反向沉淀

- [x] OneFLOW 本地 handoff、架构审计和阶段清单
- [x] 记录 persistent state、NoTrace/FullTrace、host staging 的适用边界
- [x] 记录 DTK/CMake、CTest、Slurm 证据要求和失败模式
- [ ] 脱敏整理 backend lifecycle 模板
- [ ] 脱敏整理 CPU oracle/numerical contract 模板
- [ ] 脱敏整理 HIP/DCU CMake 与 architecture probe 模板
- [ ] 脱敏整理 MPI/device mapping 与资源/退出码检查表
- [ ] 将可复用经验反向写入 `scientific-dcu-porting`

当前判定：OneFLOW 内部文档完成；经验仓库反向沉淀未完成。

## 当前执行位置

```text
阶段 0：完成
阶段 1：standalone CPU contract 完成，主 solver CPU vertical slice 待做
阶段 2：设计完成，待实现
阶段 3–5：standalone 证据完成，主 solver 集成待做
阶段 6：CPU 完成，HIP 部分完成，CUDA/KOKKOS 待验证
阶段 7：OneFLOW 文档完成，经验仓库待同步
```

