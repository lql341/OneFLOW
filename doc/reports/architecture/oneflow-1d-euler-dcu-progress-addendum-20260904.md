# OneFLOW 1D Euler DCU 移植进度补充

更新时间：2026-09-04

本补充对应 `oneflow-1d-euler-dcu-porting-checklist-20260904.md`。由于当前 sandbox 的文件更新 helper 无法读取既有 `tests/CMakeLists.txt`，新增 contract 测试目前以独立编译方式验证，尚未宣称已注册进 CTest。

## 本轮新增交付物

- [x] 新增主工程 backend-neutral domain contract：`codes/accel/include/EulerDomain.h`
- [x] contract 不依赖 `MRField`、`Zone` 或 HIP runtime
- [x] contract 固定 `CreateState → Upload → Advance → Download` 生命周期
- [x] contract 固定 `NoTrace` / `FullTrace` 运行模式入口
- [x] contract 固定 `solverIndex + localZoneId + gridLevel + backend kind` 状态 key
- [x] contract 提供问题和 field view 的尺寸/指针校验
- [x] 新增 contract 单元测试源码：`tests/euler_domain_contract_test.cpp`
- [x] 独立编译运行：3/3 通过

## 仍未勾选的门禁

- [ ] 将 contract 测试注册到现有 CTest
- [ ] 实现 CPU domain adapter 并与 OneFLOW 主 solver task 结果对照
- [ ] 在 `INIT_FLOWFIELD` 后创建主 solver state
- [ ] 在 `TimeIntegral::RungeKutta` 建立 solver-aware fast path
- [ ] 接入主 solver 的 residual、boundary、checkpoint/restart 和 output 语义
- [ ] 在目标 DCU 节点验证 HIP adapter；当前没有新增 DCU 作业证据

## 本轮验证命令

```text
g++ -std=c++17 -O3 -DNDEBUG \\
  -Icodes/accel/include -Icodes/basic/include -Icodes/project/include \\
  -IThirdParty/googletest/googletest/include \\
  tests/euler_domain_contract_test.cpp \\
  /tmp/oneflow-stage1-cpu-build/lib/libgtest_main.a \\
  /tmp/oneflow-stage1-cpu-build/lib/libgtest.a -pthread \\
  -o /tmp/oneflow-stage1-cpu-build/euler_domain_contract_test

/tmp/oneflow-stage1-cpu-build/euler_domain_contract_test --gtest_color=no
```

结果：`3 tests ... [ PASSED ] 3 tests.`
