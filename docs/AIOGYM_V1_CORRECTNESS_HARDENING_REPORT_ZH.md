# AIO-Gym V1 Benchmark Correctness and Release Hardening Report

日期：2026-07-31

基线 commit：`c88ac5a21058d815b560be87c247551bb50b3b35`

## 结论

本轮计划中的 P0 benchmark correctness、P1 release reliability 和 P2 consistency 修改均已落地，并完成全仓、真实 RL、Oracle、ONNX、八场景与 clean-wheel 验证。当前工作树包含此前未提交修改，因此本轮没有自动创建阶段 commit，避免把已有改动错误混入新提交。

## 主要完成项

- Reward identity：RewardSpec 使用完整 canonical content 计算 SHA-256；相同 ID 的不同定义会被 registry 拒绝；环境、Dataset、checkpoint 与 artifact 均携带 reward hash。
- Environment identity：`ResolvedEnvSpec` 使用 `aiogym.resolved_env_spec.v2`，hash 覆盖完整 reward 定义而不只覆盖 ID。
- Official anchors：四组 fixed-anchor artifact 升级到 v2，保存 paired calibration samples、统计量、effect size、SNR、阈值和生成环境；loader 与 ranking 双重拒绝退化 gap。
- Case repairs：修复 Cascade、Cascade Recirculating、Quadruple 的 nominal/shifted 可辨识性；Cascade economic 增加可恢复 feed-capacity loss，使 reference/bad utility 具有实质间隔。
- Final test：lock 仅在 artifact 原子提交、fsync 和 SHA-256 记录完成后进入 `complete`；evaluation 和 artifact commit failure 分阶段记录。
- Output safety：train/benchmark 默认不覆盖；run claim 阻止并发写同名 run；fresh/resume/overwrite 的 identity 与失败 provenance 明确。
- Resume correctness：checkpoint v2 保存 validation selector records、global best、验证历史、下一验证边界和 selected checkpoint hash；SB3/RLPD 在第一次 resume evaluation 前恢复；改变 `n_envs` 被拒绝；BC resume 明确不支持。
- Dependency boundaries：CI 拆为 core、RL、Oracle、ONNX、nightly E2E 和 release-wheel；新增 `test` extra 与 pytest markers。
- Public consistency：controller 必须显式提供 scenario 或 model；`list_controllers()` 返回所有注册 ID（包括 lazy Oracle）；公共 seed 入口统一使用一个 validator。
- Training budget：config 升级为 `aiogym.rl_training_config.v3`，canonical JSON 使用 `{unit, value}` budget；BC 使用 `optimizer_updates`，在线算法使用 `environment_transitions`；v2 只读迁移会记录 metadata 并发出 warning。
- Optional export：ONNX export 从 backend 移到 verified canonical checkpoint lifecycle；默认失败不破坏 native success，strict 模式在保留 native checkpoint 和 artifact 后报错。
- Dataset transaction：manifest commit 失败会删除新 shard；resume 会校验 referenced shard，并把 unreferenced/temp shard 隔离到 `shards/.orphaned/`，记录 recovery event，且可重复执行。

## Official Track / Anchor identities

| Track | Track hash | Anchor | Anchor artifact hash |
|---|---|---|---|
| cascade-economic-specialist-v1 | `b1add7956f00b4f11784a18b007f3c695a13250ff83075569a1020adf75bdc9c` | cascade-economic-anchors-v2 | `4e1bb74862d681f90bb1638d742e288817273e2c2816e731f129622f93d1c20d` |
| cascade-regulation-generalist-v1 | `e078072c8b61e68de84393af360613923fa295b729fa4b94d07a62ed95a957ca` | cascade-regulation-anchors-v2 | `c2143b13098a63068b7cdb96c7c258bc3e0a7d80d6223a1289c2f29f44635de7` |
| cascade-recirculating-regulation-generalist-v1 | `3b7c280cd79087d2731a38b7baae72d95044aa699a2f5f5d3c48321676b03fbb` | cascade-recirculating-regulation-anchors-v2 | `ea229ea4816d1d9e3efce8103ce79de2857cb01cb245611ae8d3acc98a9f1c43` |
| quadruple-regulation-generalist-v1 | `112931a0adab0e01d6d4c26c4187ad05a84d8cf1e6ac1c886350915068d284c6` | quadruple-regulation-anchors-v2 | `a41a6806a027a87a22923b631377e865c9bd540e8a23f322c4d874e90ed2db27` |

正式 anchor audit 共检查 14 个 case，全部 `quality_passed=true`。Cascade economic absolute gap 为 `12.297716002506919`，relative gap 为 `0.5647177288947097`。

## Validation evidence

- 完整测试：`483 passed`（包含沙箱外 multiprocessing、真实 SB3/RLPD、CasADi 和 ONNX Runtime）。
- Core-only clean-wheel suite：在未安装 Torch、SB3、CasADi、ONNX/ONNX Runtime 的隔离环境中 `437 passed, 11 skipped, 24 deselected`。
- Marker suites：RL `19 passed`，Oracle `11 passed`，ONNX `3 passed`。
- Quadruple/Cascade collect → BC/SAC/RLPD → resume/reload → validation E2E：通过。
- 八个 scenario 均完成 discover → make_env → reset → step，reward 全部 finite。
- `python -m compileall -q aiogym`：通过。
- Ruff：通过。
- `git diff --check`：通过。
- 隔离构建：sdist 与 `aiogym-0.1.0-py3-none-any.whl` 构建成功。
- Clean wheel venv：core dependencies 安装成功；`import aiogym` 未 eager import Torch、SB3、CasADi、ONNX 或 ONNX Runtime；Quadruple reset/step 与 `aiogym --help` 通过。

## Scope change

基线统计为 173 个 production Python 文件 / 34,718 行，以及 45 个 test 文件 / 9,017 行。当前为 178 个 production Python 文件 / 36,477 行，以及 51 个 test 文件 / 10,112 行。新增代码主要集中在 anchor quality、run claim、canonical exporter、恢复状态验证和相应 failure-injection tests。

## Release notes

- 这是 pre-v1 工作树，官方 Track ID 保持不变，anchor ID 升级为 v2；上述 Track hash 和 artifact hash 必须随 release notes 一同冻结。
- v2 RL config 可读取迁移，但所有内置 quickstart/tuning config 已改为 v3 budget；新生成的 resolved config 不再输出 `total_transitions`。
- ONNX 使用 Torch legacy exporter 时会产生上游 deprecation warning，不影响生成结果或 ONNX Runtime 验证；后续可单独迁移到 dynamo exporter。
