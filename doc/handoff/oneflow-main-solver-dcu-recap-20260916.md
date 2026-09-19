# OneFLOW 主 solver DCU 阶段 Recap

**日期：** 2026-09-16
**分支：** `dev`
**远端基线：** `origin/dev` 为 `e584bcbb`（本文写就时为 `9cc92500`）
**本地功能提交：** `f2b3d43c`，已随 `e584bcbb` 推送（本文写就时尚未推送）

## 结论

3D 主 solver 的 DCU 正确性纵切线已经跑通，且完成了公平的
**8 CPU MPI ranks vs 1 DCU** 端到端测量；但当前 host-staged 路径只达到基本持平，
还不能宣称有意义的生产加速。F 阶段仍不能整体勾选完成。

当前真正的 blocker 有两个：

1. 3D m6 50-step 在 CPU 与 HIP 共用配置下约第 20 步共同出现负压、Inf/NaN；
2. 本轮 MPI state-sync 修复的 fresh CPU 五算例门禁被 continuation fixture/runner
   阻塞：solver 返回 0，但未生成测试期望的结果文件。当前和上一版 restart fixture
   均能复现，暂不能把本轮 fresh regression 标成 green。

## 进度总览

| 阶段 | 状态 | 说明 |
|---|---|---|
| A 运行基线 | 完成 | fork、分支和验证环境已固定 |
| B standalone 1D CPU/HIP | 完成 | CPU/HIP contract、MPI 和真实 DCU 最小闭环 |
| C accelerator substrate | 完成 | runtime、backend、flux seam、device buffer |
| D domain contract | 完成 | layout、geometry、capability、state owner、生命周期 |
| E CPU vertical slice | 完成 | CPU batch、INIT/restart、RK、legacy oracle、trace |
| F DCU vertical slice | 未完成 | 1-step/3-step 正确性完成；50-step 稳定性、fresh regression、有效加速未完成 |
| G MPI/性能 | 未完成 | halo、多卡、GPU-resident 和性能优化尚未完成 |

## 已验证证据

验证环境为 Kunshan Z100 DCU、DTK 26.04、HIP architecture `gfx906`、单 DCU；
资源和模块以仓库中的 Kunshan profile 为准。

- 257 faces × 5 equations one-call：CPU/HIP flux 最大差 `6.661e-16`，
  residual 最大差 `1.110e-15`。
- 小 case 1-step LU-SGS：HIP 对 legacy 的 invflux/residual 最大差
  `1.4210854715202004e-14`，state 最大差 `6.514681130330882e-19`；
  finite、density/pressure positive。
- 小 case 物理语义：residual 重建误差 `0`，全域 residual/边界通量闭合误差
  `2.804e-13`；boundary、内部面守恒、ALE、face-area contract 均通过。
- 3D m6，896256 faces，3-stage RK，1-step：HIP 对 legacy 最大差
  `1.7064127888488656e-13`，state finite，density/pressure positive。
- 3D m6，3-step：36 条 legacy/CPU batch/HIP batch trace 全通过；HIP 对 legacy
  最大绝对差 `2.8399504969911504e-13`，最大 scaled error
  `1.992850329202156e-13`。
- root HIP smoke、GoogleTest、hardware HIP CTest 和 stage benchmark 均通过；
  测试集合非空，失败会传播为非零 workload/Slurm 状态。

## 公平性能数据

测量使用同一输入、同一配置和同一 `steps=3,warmup=1,repeats=3` basis。
计时为端到端 wall-clock，包含 MPI launch、初始化、host pack、H2D、kernel、D2H
和输出，不能等同于 kernel-only speedup。

| 模式 | 资源 | mean lifecycle (ms) | 相对 8-rank legacy |
|---|---|---:|---:|
| legacy CPU | 8 CPU MPI ranks | 25904.228096 | 1.000000x |
| CPU batch | 8 CPU MPI ranks | 26794.339157 | 0.966780x |
| HIP/DCU batch | 1 DCU | 25658.172501 | 1.009590x |

解释：1 DCU 相对 8-rank CPU 仅快约 `0.95%`，在当前运行波动范围内，结论是
“基本持平”，不是可对外宣传的加速数据。CPU batch 反而慢约 `3.32%`，说明当前
主要问题不是简单地把 CPU batch 换成 HIP，而是需要减少 host/device 往返和同步。

## 本轮代码与 runner 变化

- `SyncAllEulerDomainStates` 改为遍历 `ZoneState::localZid`，避免 MPI 非 owner rank
  访问空的全局 grid；这是 8-rank CPU benchmark 能够启动的必要修复。
- F2 stage runner 增加 CPU/HIP 独立 MPI rank 参数。
- F3 benchmark runner 改为申请 8 tasks × 1 CPU，并支持显式记录 CPU/HIP rank 数，
  从而可以真正执行 8 CPU ranks vs 1 DCU。
- 代码和文档已在本地提交 `f2b3d43c`；当前 `origin/dev`（`e584bcbb`）已包含该提交，并已把 `upstream/master` `fa3f3b06` 合入 `dev`。

## 下一步 TODO

### P0：先恢复可重复回归

- 修复 continuation case 的 restart/output fixture 或标准 runner 的 case 隔离逻辑。
- 重新跑本轮修改对应的 CPU 五 case：
  - normal `1e-8`；
  - strict `1e-15`；
  - 5/5 case、作业状态和 workload exit code 均通过。
- 保留此前同 revision 的历史 normal/strict 5/5 证据，但不替代本轮 fresh gate。

### P1：收口 3D 物理稳定性

- 检查 CFL、初始条件、边界条件、restart 语义和 RK stage 更新。
- 先让 CPU legacy 在 50-step 稳定，再要求 CPU batch/HIP batch 与其逐步对齐。
- 继续检查 finite、positive density/pressure、守恒和 boundary semantics。

### P2：再做真实加速

- 对端到端时间分解 MPI launch、初始化、pack、H2D、kernel、D2H、输出。
- 复用 device buffer 和 device-resident state，减少每 stage 的 host/device 往返。
- 再做 device-side reduction、MPI/halo、多卡 mapping 和 GPU-aware MPI 探测。
- 只有正确性、fresh CPU regression、MPI correctness 都通过后，才发布加速结论。

## 当前 Definition of Done

- [x] 3D m6 1-step 与 3-step CPU/HIP trace 正确性。
- [x] root HIP contract、smoke、hardware CTest。
- [x] 8 CPU ranks vs 1 DCU 公平测量。
- [x] MPI rank-local state-sync 修复。
- [ ] fresh CPU normal/strict 五 case。
- [ ] 3D m6 50-step 稳定性。
- [ ] 完整粘性 NS/HIP。
- [ ] MPI、多卡、GPU-resident 性能优化。
- [ ] 有意义且可复现的端到端加速。
