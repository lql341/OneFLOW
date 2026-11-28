# OneFLOW Living TODO

本页跟踪 [Phase Plan](oneflow-development-plan.md) 的当前切片。父项和子项沿用计划编号；更细的执行步骤记在对应子项下。当前验收完成后移出本页，将复验结论和证据索引留在计划或对应报告。

## 当前状态（2026-11-23 收口）

分支 `dev`，HEAD `813d364ed`，即 upstream master 合并 PR #267 后的提交；`dev`、`origin/dev`、`origin/master` 与 upstream master 完全一致，工作树干净。PR #267 与上游 master 推送的 CI 均全绿。

**文档根改名遗留已收口（2026-11-23）**：修掉 `PublishMySite.yml` 的构建路径（原 `cd doc`），清掉 53 处 `doc/` 残留，其中 `stl_geometry_asset_preparation.json` 的 7 处曾让 `test_contract_catalog` 在 master 上静默红 8 个用例。新增 `scripts/check-doc-paths.py`，接入 `docs-check.yml`（仅 master；hexin 仍用 `doc/` 根且有自己的存量引用，合并后再加）与 Linux CI 的契约测试步骤。根因与坑记录见[当前交接](../handoff/oneflow-current-handoff.md) §5。

## 上一个收口点（2026-11-22）

分支 `dev`，HEAD `4dac26c68`（已把 upstream master `a307becac` 并入）。旗舰算例 `test/cfd/m6wingroe_sa`（vismodel=3 SA / Roe ischeme=1 / LU-SGS time_integral=2 / ilim=0 / 单 zone / ivischeme=0）在昆山单 rank 已跑通**完整 GPU 通量路径 + 完整 SA 链路**。固定路径、构建命令见[当前交接](../handoff/oneflow-current-handoff.md)，架构细节见[3D device residency 设计](../architecture/oneflow-3d-device-residency-design.md)。

**合并注记（2026-11-22）**：`master` 已三端同步到 upstream `a307becac` 并并入 `dev`（`4dac26c68`）。该合并只带来 R1 运行时与几何资产相关内容，**未触及**本页跟踪的加速路径（`codes/accel`、`codes/uturb`、`ci/kunshan` 均无改动），因此下方结论与证据不受影响。

**合并后验证已通过**（`evidence/gate/128698741-verify9f9`，HEAD `9f9d67e81`）：configure / runtime objects / link 全部 OK，两臂 rc=0；**aero max_rel=0 逐字节一致**，residual `1.808152e-12` 与合并前完全相同（即合并未引入回归，该值仍是 B.4.1 的既有问题）；同作业两臂 20 步 74.93s → 32.8s（2.28x）。

### 已达成（准确性）

- **气动力 aero.dat 逐字节一致**（全部 20 步，max_rel=0）；主控残差 res.dat max_rel `6.78e-21`（gate 1e-12 pass）；湍流残差 turbres 末位 `1.8e-16`；当前流场 flow.dat level1 `2.7e-15` 绝对；湍流场 turbflow 全层 `1.4e-17`；可视化 visual.dat 差异全为机器零（1e-37）。证据：`xb-dlfinal/128674985`、`xb-final-check/128676813`。
- **HIP 重跑确定性**：同 binary 同配置两次，aero/res/turbres 逐字节相同（`xb-determinism/128677591`）。
- **SA 全链路设备化后数值等价**：aero 仍 max_rel=0；主控残差与湍流残差与移植前**逐位相同**（`1.808152e-12` / `8.168e-14`）。证据：`128693499-saport`、`128694886-saport`。
- **flow.dat level2（历史槽）差 7.58e-3**：= 一步物理变化，HIP `InitUnsteady` 重建历史层 vs CPU 保留上一步；restart reader 用 level1 重建 current，**不影响续算**（`xb-history-shift/128676991`）。

### 已达成（性能，20 步单 rank 实测）

所有数字都取**同一次作业、同一节点**的 CPU/HIP 两臂对照；跨节点比较不成立（实测同配置不同节点墙钟可差 8%，见下）。

- 端到端 wall **77.09s → 35.19s（2.19x）**（`128693499-saport`，节点 b13r2n04）。
- 通量层 rhs **7.9x**（无粘 23x、粘性 3.9x）；迭代段 1.40x。
- **boundary BC device 化：~20000ms → ~90ms（~220x）**，aero 仍 0（`xb-boundary/128680850`）。
- **SA 链路 device 化：`update_residuals` 33.8s → 18.5s（1.83x）**、`calc_boundary` 2.1s → 1.1s。

### 已知不达预期（实测，勿再按旧结论推理）

- **LU-SGS 设备化比 host 慢**。同运行同节点对照（`128693499`，b13r2n04）：CPU 臂上下扫合计 **43.05s**、HIP 臂 **45.91s**，慢 6.6%；去掉无用上传后的复测（`128694886`）HIP 44.4s，仍慢于同节点 CPU 的 43.9s。根因是网格 48×48×128 的依赖层级约 221 层、每层仅约 1300 单元，每步 2×221 次内核启动的开销超过并行收益。当前按用户决定**保留启用**，是下一个要处理的大项。
- **门禁口径陷阱**：`f2-main-solver-cross-backend-residual.sh:246` 的相对差是 `|d|/max(1,|a|,|b|)`；残差量级 1e-6，故 "1e-12 门禁" 实际是 **1e-6 相对**。`residual 1.808e-12 fail` 对应真实相对差 **5.3e-7**（step20），不是"接近机器精度只差末位"。
- **`tasks.tsv` 时间是真实值的 2 倍**（`CmxTask.cpp:483` 与 `Command.cpp:295` 各记一次），`stage.tsv` 才是单次。对比两组数字前必须确认口径，否则会得出错误的加速比。
- **节点间性能波动可达 8%**：同一份代码 20 步墙钟 b13r2n04 为 35.19s、b13r3n03 为 38.06s，两臂同步变慢（CPU 77.09→83.42s），是节点差异不是代码回归。**任何加速比必须取自同一次作业的两臂**。

## 进行中 / 待收口

- [x] **B.4.1 boundary 残差 1.808e-12**（**Done**，三级验收全部通过）
  - [x] **L1 host-flux 20 步**：`evidence/gate/128965585-b41sym`，residual/aero 均 `max_rel=0`。当时两臂都不走 device flux，所以这一级只隔离了 boundary 一项。
  - [x] **L2 full-dev 20 步**：`evidence/gate/129045272-b41sym`，`DEVICE_FLUX_ON=yes`、`DEVICE_GRADIENT_ON=yes`、`TIMING_USABLE=yes`（parse exit 0），residual `6.776264e-21` pass、aero `0`。
  - [x] **L3 full-dev 200 步**：`evidence/gate/129048798-b41sym`，同三判据全 yes，residual `3.218725e-20` pass（row 179 field 6）、aero `0`，两臂 rc=0。`flow` 仍是 level2 历史槽 `binary_byte_diff`，属已知项，不影响续算。
  - [ ] **L4 8 zone / 8 rank（未完成，二分已定位）**：上述 L1–L3 都是 **1 zone / 1 rank**。8 zone 的 flagship full-dev 数值 fail，已用三组实验锁定问题在 device 梯度链，不在 device flux：全 host `129082789` residual 0 pass；device flux on + 梯度 off（原始代码）`129083701` residual 3.4e-21 pass；双 on `129056842` residual 2.0e-6 fail。另注：8 zone 下 aero 3.6e-2 的差在纯 host 对照里同样存在（分区累加顺序差异），B.4.1 的 aero 闸门只对单 zone 有效。一个「强制全量梯度下载」的假设被证伪（单测 24/24 但集群 residual 恶化到 1.4e-4），已回退，详见 handoff §5。修好 device 梯度的多 zone 正确性后才能谈 8:1 加速比。
  - [x] 去掉 `HipBoundarySupported` 的 `vismodel!=0` 硬拒绝；kernel 补 viscous 壁面、symmetry ghost、bcQ 覆盖所有 BC。提交 `ab7a40fa8`。
  - [x] `bnd_kernel 2.78 / bnd_ghost_copy 10.04 / boundary_device 89.19ms`，`boundary_calc_face_bc` 归零；aero max_rel=0。
  - [x] **已排除 SA 移植的影响**：SA 设备化前后 res/turbres 逐位相同，证明这 1.808e-12 来自 NS 非壁面 BC，与湍流无关。
  - [x] **已收：对称面 ghost 反射语义**（2026-11-26）
    - 假设成立。host `NsBcSolver::SymmetryBc` 把 `primt1` 的速度分量覆盖成反射值，`UNsBcSolver::UpdateBc` 写回 `q[rc]`；device type==3 却写 owner 副本。ilim=0 时 ghost 进 `0.5*(q[lc]+q[rc])`，对称面法向速度没被消掉 → 通量不同 → 残差。
    - 改法：新增 backend-neutral `codes/accel/include/BoundaryArithmetic.h`（`ReflectVelocityRelativeToMovingFaceHd` / `ArithmeticAverageHd` / `RelativeFaceNormalVelocityHd`，仿 `TurbSaArithmetic.h` 的 HD 宏），host `NsBcPolicy.cpp` 与 device kernel 都转调同一实现，公式不再有第二份转录。
    - 验收：`evidence/gate/128965585-b41sym`（20 步）、`evidence/gate/128967034-b41sym`（200 步），两作业均 `residual max_rel=0.000000e+00`、`aero max_rel=0`，两臂 rc=0。残差非平凡（step20 `5.01829618388647769e-06`，step200 `3.06957041038148102e-06`），HIP 臂 `boundary_device 97.77ms` / `bnd_kernel 2.82ms` 非零证明设备真执行。
    - 本地契约测试：`bc_policy_test` 新增 3 例（法向抵消、面平均法向归零、斜法向幅值守恒），13/13 通过。
    - **归因已隔离**：只含 `-O3`、不含 symmetry 修复的快照独立跑 20 步门禁（`evidence/gate/128976121-o3only`，binary `4b8b4ef9`），residual 仍为 `1.808152e-12` fail、aero 仍 0。因此 `-O3` 对数值无影响，逐位归零完全来自 symmetry ghost 修复。
- [x] **B.4.1 次要项：farfield 幂函数快路不一致**（Done）
  - host `FarFieldPower` 的 `x*x*sqrt(x)` 快路只在 `vismodel == INVISCID` 时走；device `BoundaryPower` 原先无条件走。已给 device 加 `inviscid` 参数（= `!viscousWall`，等价于 `vis_model.vismodel == INVISCID`），三处调用点同步更新。随同上条 200 步门禁一并验证为逐位一致。
- [ ] **B.4.2 SA 湍流 device 化**（Done，code-complete）
  - [x] 梯度（CSR 重写，去掉 O(nCells×nFaces)）、显式更新、隐式谱、边界、LU-SGS 全部接线；`TurbSaGradientView`/`TurbSaUpdateView` 由孤儿代码变为生产路径。
  - [x] 三个 "已有入口但从未调用" 的根因已修：`CalcGradient`/`Update` 曾把 host 指针直接传给 kernel。
  - [x] 集群验证：编译链接通过，HIP rc=0，数值与移植前逐位等价。
- [ ] **B.4.3 HIP 开关语义统一**（Done）
  - [x] 9 个 `ONEFLOW_ENABLE_UTURB_HIP_*` 并入 `HipProductionPolicy.h` 白名单，backend=HIP 即生效；显式 =0 仍可单独关闭。与 UNS 语义一致。
- [x] **完整 device 链 env 硬化**（Done，B.4.4-prep 前置）
  - [x] 根因：`vismodel != 0` 时 `UseHipBatchAdapter` 与 `UseHipGradient` 各查一个独立开关
    （`UNsInvFlux.cpp:899` / `:791`），只设前者会得到「device flux + host 梯度」的混合路径
    （作业 `128983927`：`gradient_kernel=0`、`rhs_inviscid` 360 ms/step）。
  - [x] `ci/kunshan/f2-main-solver-cross-backend-residual.sh`：flagship profile **默认双 allow**，
    inviscid profile 保持 0，两者都可用 `=0` 覆盖出 host-flux 对照臂。
  - [x] 新增 `ci/kunshan/parse-xb-device-path.py`：三判据（`DEVICE_FLUX_ON` / `DEVICE_GRADIENT_ON` /
    `TIMING_USABLE`），exit 0=完整设备、exit 2=mixed 或 host_fallback；gate 末尾自动输出 `device-path.txt`。
  - [x] `tests/euler_inv_flux_capability_test.cpp` 补 2 例（viscous 与 opt-in 无关、ilim=0 仍需 5 方程），8/8 通过。

## 后续（性能提升路径）

- [ ] **B.4.4 LU-SGS（full-dev 基线已建立，待决策）**
  - **旧结论作废**：「device LU-SGS 比 host 慢 6.6%」不成立。主控 LU-SGS 至今**没有设备代码**
    （`UNsLusgs.cpp` / `NsLusgs.cpp` 的 `Hip` 引用数为 0），所谓差异来自不同轮次的 env 与步数口径不一致。
  - **full-dev 200 步基线**（`evidence/gate/129048798-b41sym`，b13r3n03，1 rank，binary `c51e1706`）：
    单步顶层合计 CPU 7155.56 → HIP 2957.66 ms/step，**端到端 2.42×**。
  - **口径警告**：上述 2.42× 是 **1 核 / 1 DCU**，与历史 18.5×（8 核 / 1 DCU 的无粘受限路径）不可比。
    1 核跑 294,912 单元时 cache 几乎全失效，CPU 基线被高估；8 核分摊后每核 36,864 单元进 L2，
    从 1→8 核墙钟改善达 18.6×（超线性）。8c/8c1d 口径下新测的 device flux 单臂
    `rhs_inviscid` 仅 0.76×（`129083701`），远小于 1c/1d 的 0.03×。在 8 zone device 梯度修好、
    且用同 rank 数配对测出可信 8:1 数字之前，不要把 2.42× 当作对外性能结论。

    | 顶层阶段 | CPU ms/step | HIP ms/step | HIP 占比 |
    |---|---:|---:|---:|
    | LU-SGS 上下扫 | 1864.82 | 1558.92 | **52.7%** |
    | update_residuals | 3159.49 | 1161.33 | 39.3% |
    | calc_time_step | 196.44 | 164.26 | 5.6% |
    | calc_boundary | 1894.54 | 37.53 | 1.3% |
    | lusgs_update_flowfield | 40.27 | 35.62 | 1.2% |

  - **LU-SGS 仍是 host**：HIP/CPU = 1558.92/1864.82 ≈ **0.84**（并非加速）。对照
    `rhs_inviscid` 0.03、`reconstruction` 0.01、`calc_boundary` 0.02——真正上设备的三项都是 1–2 个数量级。
  - **它现在是 HIP 侧最大的单项**（52.7%），也是唯一没上设备的大块。
  - **next action 三选一（只能选一条）**：
    - (a) 维持 host LU-SGS，继续优化仍走 host 的 `rhs_viscous`（HIP/CPU 0.78，575.37 ms/step，占 19.5%）与同步开销（B.4.5）
    - (b) 立项**主控 LU-SGS device 化**（单 zone 优先，独立 milestone；当前 0 设备代码，收益上限约 52.7% 的顶层时间）
    - (c) 先做 **A.4 宏步**，减少每步 LU-SGS 前后的 H2D/D2H 往返
  - **建议**：(b)。理由——(a) 的剩余空间只有 `rhs_viscous` 的 22% 与同步项，(c) 的前置是 A.4 本身尚未实现；
    而 (b) 直接作用在 52.7% 的最大项上，且主控 LU-SGS 是标量 5 方程块扫、比 SA 的单方程扫更适合按层并行。
    立项前必须先做一次**层数测量**（历史记录 221 层、每层约 1300 单元），若层数确实上千，
    每层一次 kernel launch 的固定开销会吃掉并行收益——这正是当初"设备更慢"传闻的来源，
    需要先把这个数字测准再决定是否值得做。
- [ ] **B.4.5 同步开销**：`gradient_download 206ms / v_download 116ms / v_upload 88ms / gradient_mirror 37ms`（20步累计）是为 oracle 对齐付出的回传代价，SA 常驻化后可减少。
- [ ] **B.4.6 更长步数/多 rank**：35.19s 是 20 步含固定开销的数字，需 200+ 步、多 rank 复测端到端。

## 证据索引

- 准确性 gate：`evidence/gate/xb-dlfinal/128674985`（res 6.78e-21, aero 0）
- 三项输出：`xb-final-check/128676813`、`xb-history-shift/128676991`
- 确定性：`xb-determinism/128677591`
- boundary device：`xb-boundary/128680850`（220x，res 1.8e-12 待收）
- **B.4.1 收口（symmetry ghost）**：`evidence/gate/128965585-b41sym`（L1 20 步）、`evidence/gate/128967034-b41sym`（200 步）
- **B.4.1 L2/L3（full-dev）**：`evidence/gate/129045272-b41sym`（20 步，三判据 yes）、`evidence/gate/129048798-b41sym`（200 步，residual 3.2e-20 / aero 0）
- **B.4.4 full-dev 基线**：同 `129048798`，LU-SGS 占 HIP 顶层 52.7%，端到端 2.42×
- **混合路径反例（勿当基线）**：`evidence/gate/128983927-b41sym`，只设 batch allow，`gradient_kernel=0`
- **SA 全链路**：`128693499-saport`（首版，b13r2n04）、`128694886-saport`（去无用上传后，b13r3n03）
- 早期粘性：§5.40–§5.44；H1/H2：§5.2；网格阶梯：§5.6/§5.9

## 接手第一条命令

集群作业脚本已经在位，不用重写：

```bash
# 1) 打包当前工作树（未提交改动也要进包）
cd <repo> && SHA=$(git stash create) && \
  git archive --format=tar.gz --prefix=OneFLOW/ "$SHA" -o /tmp/of-sa.tgz

# 2) 上传（超时常见，重试；成功后核对字节数）
python3 ~/.agents/skills/scnet-hpc/scripts/scnet.py --region 11250 \
  upload /tmp/of-sa.tgz /public/home/luql/oneflow/snapshots --cover

# 3) 提交：脚本内解包→构建 RuntimeObjects+fast→跑 20 步 CPU/HIP gate
python3 ~/.agents/skills/scnet-hpc/scripts/scnet.py --region 11250 submit \
  --name of-saport --command "bash /public/home/luql/oneflow/scripts/of-saport.sh" \
  --work-dir /public/home/luql/oneflow/logs --queue kshdnormal \
  --nodes 1 --cpus 8 --dcus 1 --memory 27G --walltime 02:00:00

# 4) 等结果，evidence 落在 evidence/gate/<jobid>-saport/
python3 ~/.agents/skills/scnet-hpc/scripts/scnet.py --region 11250 wait <jobid>
```

注意：**memory 上限 27G**（32G 会被 DefMemPerCPU 拒）；`of-saport.sh` 末尾已经把 `cross-backend.tsv`、两臂 stage 表、HIP 日志尾都打进 `result.txt`，不必单独取。

[历史 Living TODO 快照](https://github.com/lql341/OneFLOW/blob/1eee55499a69e6b9bc7dadf28188f484af0fc983/docs/plans/oneflow-development-todo.md)保留原章节、测量口径和证据位置。
