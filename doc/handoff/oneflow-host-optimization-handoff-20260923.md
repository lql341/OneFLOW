# OneFLOW host 优化交接 TODO

**日期：** 2026-09-23  
**适用分支：** `dev`  
**文档性质：** fork-only 工作交接记录，不是正式性能报告

## 先看结论

当前工作已完成一轮 host buffer/lifecycle 优化，并在昆山目标节点闭环验证：

- CPU normal/strict 五算例均 `5/5`。
- Kunshan DTK 26.04、`gfx906`、单卡 DCU 的 root HIP build 成功。
- m6 同 basis 完整 workload 三次正式重复均 `exit_code=0`，Slurm 均 `COMPLETED/0:0`。
- 最新源码版本的探索性 wall-clock：
  - legacy CPU 8 ranks：`4834.289 ms`
  - HIP 1 rank + `dcu:1`：`1855.713 ms`
  - 单次作业 ratio：`2.6051x`

该 ratio 是完整任务 wall-clock，包含初始化、MPI 启动、host staging、H2D/D2H、kernel、同步和状态更新，不是纯求解器 kernel 时间。由于各次 benchmark 使用不同作业/节点，不能把跨作业差值直接视为稳定收益；正式性能报告暂不更新。

## 当前代码已经做了什么

### Host lifecycle reuse

- `NsCalcInvFlux()` 使用 function-local static `UNsInvFlux`，跨 RK stage 复用对象。
- `NsLimField::Init()` 在方程数和 face extent 不变时复用 `qf1/qf2`。
- `UNsInvFlux::Alloc()` 在方程数和 face extent 不变时复用 `invflux`。
- topology/extent 变化时自动释放并重新分配。
- `UNsInvFlux` 析构函数释放持久化 `invflux`。
- `DeAlloc()` 不再每个 stage 删除 `invflux`。

### HIP reconstruction metadata cache

- `UNsInvFlux` 持久保存 `boundaryOperation` 和 `bcQ` host 容器。
- boundary operation 依据 grid、`BcRecord`、face extent、boundary-face extent、equation extent 失效。
- `HipGradientStorage` 记录 host operation pointer/extent，只在变化时重传静态 operation metadata。
- `bc_q` 数值仍每个 stage 更新并上传，不能因为缓存而跳过，避免改变 solid boundary 语义。

### Host boundary/field loop cleanup

- `NsCalcGamaT()` 改用连续 `data()` 指针访问 density、pressure、gamma、temperature。
- `FarFieldBc()` 已有 reference sound speed/entropy cache。
- inviscid、`gamma≈1.4` 的 FarField 幂运算使用 `x*x*sqrt(x)`；viscous 和其他 gamma 保持原始 `pow()`，CPU strict oracle 不改变。
- boundary profiler 默认关闭时走无计时分支；开启时按 region 累计，不在每个 face 调用 `getenv`/mutex。

## 关键源码位置

- [host inviscid lifecycle](../../codes/ns/src/NsRhs.cpp)
- [host limiter buffers](../../codes/uns/src/UNsLimiter.cpp)
- [inviscid flux lifecycle/reconstruction adapter](../../codes/uns/src/UNsInvFlux.cpp)
- [inviscid flux state/header](../../codes/uns/include/UNsInvFlux.h)
- [HIP metadata cache/backend](../../codes/accel/include/HipFluxBackend.h)
- [HIP reconstruction implementation](../../codes/accel/src/HipFluxBackend.hip)
- [FarField host fast path](../../codes/ns/src/NsBcSolver.cpp)
- [boundary profiler path](../../codes/uns/src/UNsBcSolver.cpp)
- [living TODO](../plans/oneflow-development-todo.md)
- [Kunshan workflow](../../ci/kunshan/README.md)

## 已完成验证

### CPU oracle

使用标准五算例 normal/strict 回归：

- normal tolerance：`1e-8`
- strict tolerance：`1e-15`
- normal：5/5，最大绝对差 `4.970574442764598e-10`
- strict：5/5，最大绝对差 `1.1072414686508214e-17`

### Kunshan target-node benchmark

固定口径：

```text
case=m6wingroe_sa
steps=3
warmup=1
formal_repeats=3
legacy_cpu_ranks=8
hip_ranks=1
dcu=1
limiter=off
viscous=off
CFL=0.01
```

最近三次探索性版本均 workload 成功。可比较的均值如下：

| 版本 | legacy CPU mean | HIP mean | 说明 |
|---|---:|---:|---|
| host buffer reuse | `5500.673 ms` | `2286.562 ms` | cache 前基线 |
| boundary operation cache | `—` | `1968.529 ms` | 跨作业约低 13.9%，需 paired repeat |
| 最终源码版本 | `4834.289 ms` | `1855.713 ms` | 单次作业 ratio `2.6051x` |

不要把表中跨作业差值当成稳定加速结论。最终版本的 3-step HIP breakdown 仍显示：

- `boundary_gamma_inner` 约 `73 ms`
- `boundary_bc` 约 `~110 ms`
- `reconstruction` 约 `100 ms`
- `update_residuals` 约 `510 ms`

因此当前更值得继续做的是 boundary host face loop 和 residual/device residency，而不是继续微调 gamma 指针访问。

## 下一步 TODO（按优先级）

### P0：先做 paired benchmark

- [ ] 在同一个 Slurm allocation 内连续运行 A/B 两个可执行版本，避免节点和启动噪声。
- [ ] A：host buffer reuse 版本。
- [ ] B：boundary operation cache + gamma pointer 版本。
- [ ] 保持完全相同的 `steps/warmup/repeats/ranks/GRES`。
- [ ] 只在 paired 数据方向一致后，才把 cache 收益写成性能结论。

### P1：继续优化 host boundary loop

- [ ] 盘点 `UNsBcSolver::SetId/PrepareData/UpdateBc` 的 face 级字段访问。
- [ ] 预计算并缓存 region 的 face list、bc type、bc name id、left/right cell、几何量指针。
- [ ] 将 boundary operation 按 boundary type 分组，减少每个 face 的重复分支和索引查找。
- [ ] 评估 `PrepareData()` 中 equation-major field pointer 缓存；保持与 CPU oracle 相同的读取顺序。
- [ ] 评估普通 boundary、interface/periodic、solid-surface 分路径；不要改变 boundary-first 和 ghost ownership 语义。
- [ ] 重新跑 CPU normal/strict 及 3-step accuracy gate。

### P2：评估 device boundary/ghost update

- [ ] 明确 `bc_q`、ghost `q`、gamma/temperature 的读写契约。
- [ ] 先做小 case contract：ordinary boundary、interface、periodic、solid-surface。
- [ ] 保留 CPU boundary oracle 和 opt-in HIP fallback。
- [ ] 不要在没有 boundary semantics/physicality contract 的情况下直接迁移完整 `NsCalcBc`。

### P3：继续减少 host staging

- [ ] 研究 residual 初始化/`LOAD_RESIDUALS` 是否可以直接进入 state-owned device residual。
- [ ] 研究 boundary 更新后的内部 `q` 是否必须每个 RK stage 完整 D2H。
- [ ] 继续保持 `q → gradient → reconstruction → flux/residual` 的 device residency 方向。
- [ ] 记录每个阶段的 allocation、H2D、D2H、kernel launch 和 synchronize 次数。

## 不能越界的能力边界

当前 HIP 主 solver 仍只验证以下受限条件：

- 单 zone、finest grid
- 5 equations
- Lax-Friedrichs
- limiter off
- inviscid
- 单节点、单 DCU

以下内容尚未完成或不能据此宣称已支持：

- viscous/turbulence 主路径
- 多 zone、interface/periodic MPI halo
- 多卡/跨节点 MPI
- 完整 device-resident `q`
- GPU reduction
- 正式性能报告中的稳定加速结论

## 交接时的操作规则

1. 先读 [AGENTS.md](../../AGENTS.md)、本文件和 [living TODO](../plans/oneflow-development-todo.md)。
2. 不要覆盖工作区中与本轮无关的 dirty 修改；当前 `dev` 不是干净分支。
3. 先做 `git diff --check`，再改代码。
4. 性能比较必须写清 `steps/warmup/repeats/ranks/GRES` basis。
5. CPU 或 HIP 变化必须先过对应 correctness gate，再解释 timing。
6. 不把 Slurm 原始日志、绝对路径、主机名、作业号或账号写入提交文档。
7. 正式性能报告的 Markdown/HTML 暂不更新；探索性数据只写 living TODO 或本 handoff。

## 完成后回写

下一次工作结束时，至少更新：

- 本文件的“当前代码已经做了什么”；
- “下一步 TODO”复选框；
- `doc/plans/oneflow-development-todo.md` 的日期条目；
- 若有正式报告资格，再同时更新对应 Markdown 与 HTML。

