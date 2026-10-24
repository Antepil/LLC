# Neural ODE Simulink 双支路对比模型设计

## 1. 目标与动机

新建一个可打开、可仿真、可生成截图的 Simulink 模型，在同一次仿真中并行运行参考 LLC 功率电路和训练后的 Neural ODE。两条支路共享 `Vin` 与 `Vref`，但各自维护独立的 PI/PFM 控制器状态；模型直接输出参考状态、预测状态、状态误差、两套门极和两套控制量。

该模型用于直观检查闭环结构和波形，不替代现有 MATLAB 验证入口的指标与报告流程。首版成功标准是结构可复现、运行时模型注入明确、冷启动和预热状态交接可执行，并能用自动化测试证明连续 Neural ODE 状态确实由 Simulink 求解器推进。

## 2. 已选方案与备选方案

### 已选：同模型双支路 + 连续 Level-2 MATLAB S-Function

从现有 `llc_full_bridge_closed_loop.slx` 派生新模型 `llc_neural_ode_comparison.slx`。保留参考电路支路，增加 Neural ODE 连续状态块、第二套独立 PI/PFM、状态误差和日志模块。Neural ODE S-Function 暴露四个连续状态，导数由已导出的网络计算，积分和切换事件交给 Simulink 求解器。

选择原因：它与当前 `ode23t` 和精确 PFM 事件语义最一致，并允许在同一 Scope 和同一时间轴上直接观察两条支路。

### 未选：Neural ODE 单独模型

结构更简单，但需要两次仿真后再对齐，不能满足“同一模型中看到参考与神经网络”的目标。

### 未选：固定步长离散 Neural ODE

容易实现，但会额外引入离散积分误差，首版无法区分网络误差和离散求解误差。

## 3. 模型结构

```text
                         +--> 参考 PI/PFM --> 参考门极 --> LLC 功率电路 --> x_ref --+
Vin / Vref 共享激励 ----+                                                        +--> x_ref - x_nn
                         +--> Neural PI/PFM --> 神经门极 --> Neural ODE --> x_nn ---+
```

- 参考支路沿用现有 Simscape LLC 电路与 `llc_closed_loop_pfm`。
- Neural ODE 支路使用独立的 `llc_closed_loop_pfm` 实例，反馈自己的 `v_o`，不得读取参考支路输出。
- 两个控制器接收相同 `Vref`，但 DWork 状态、门极时序、频率和积分量完全独立。
- Neural ODE 输入顺序固定为 `[i_Lr, v_Cr, i_Lm, v_o, Q1, Q2, Q3, Q4, Vin]`，与模型导出契约一致。
- 模型记录 `llc_states`、`llc_neural_states`、`llc_state_error`、`llc_gates`、`llc_neural_gates`、`llc_control_log` 和 `llc_neural_control_log`。
- 模型提供参考/预测状态 Scope 和误差 Scope，块名和信号名保持稳定，便于脚本和截图定位。

## 4. Neural ODE 连续块

新增 `llc_neural_ode_sfun.m`：

- 一个 5 维输入端口：四路门极和 `Vin`；一个 4 维输出端口：物理状态。
- 四个连续状态，由 Simulink 连续求解器维护。
- `Derivatives` 使用现有 `llc_neural_ode_forward` 完成归一化网络前向，再按 `state_scales/time_reference_s` 转回物理导数。
- 运行参数为一个标量结构 `llc_neural_runtime`，至少包含 `model` 与 `initial_state`。
- `model` 必须通过现有加载器契约；状态、门极、Vin 或网络输出非有限时显式抛出 `llc:*` 错误。
- `.slx` 不固化训练权重。保存模型只保留带明确占位标记的运行参数；未注入真实或测试模型时，仿真必须失败，不能静默运行占位网络。

## 5. 控制器与启动状态

现有 `llc_closed_loop_pfm` 仍使用一个对话参数。为保持旧模型兼容，参数既可为原控制配置，也可额外包含 `initial_state`：

- 无 `initial_state` 时保持当前冷启动行为不变。
- 有 `initial_state` 时，显式恢复相位、积分、滤波电压、频率、参考、电压误差、活动桥臂、待导通桥臂以及下一控制/边沿/导通事件。
- 传入的事件时间采用当前仿真的局部时间；预热交接调用方负责把绝对时刻平移到零点。
- 控制器状态同时保存 `reference_ramp_elapsed_time_s`。冷启动为零；预热启动恢复 handoff 时已累计的斜坡时间。控制更新使用“恢复值 + 当前局部时间”，因此 warm 模式不会把已经升起的 `Vref` 错误地重新拉回零。
- 两条支路在预热交接瞬间接收数值相同但存储独立的控制器状态，之后分别由各自反馈演化。

运行入口支持：

- 冷启动：四状态为零，控制器使用现有默认初始化。
- 预热启动：沿用现有独立闭环语义，先只运行参考电路到 handoff，确认稳态并重建完整控制器状态；评估起点把参考 handoff 四状态同时复制给参考电路和 Neural ODE，两套控制器也从相同的完整交接状态各自恢复。首版不单独预热 Neural ODE。缺字段、非有限值或过期事件时间必须报错。

首版不在 `.slx` 内执行预热阶段。现有预热仿真与稳态判定仍在 MATLAB 驱动层完成，评估阶段才启动双支路模型。

## 6. 构建与运行接口

- `build_llc_neural_ode_comparison_model(modelDirectory)` 从闭环参考模型复制并生成目标 `.slx`，重复执行得到相同块名、连线和日志接口。
- `llc_simulate_neural_ode_comparison(...)` 负责加载目标模型、注入硬件/求解器/激励、网络、两条支路初始状态和控制器状态，运行后返回统一时间轴上的参考、预测、误差、门极和控制量。
- `llc_run_neural_ode_simulink_comparison(configPath, experimentName)` 是新的顶层入口：复用现有配置加载器及 cold/warm 实验定义，完成必要的预热后调用上述仿真函数。配置在 `paths.simulink_comparison_model_file` 指定新模型路径；缺失字段时显式报错。
- 构建器只负责结构，不读取 checkpoint；运行入口只接受已经由 `llc_load_neural_ode_model` 校验的模型结构。
- 现有 `llc_run_neural_ode_validation` 的 MATLAB 独立积分后端保持不变，避免在首版中改变已经验证的结果路径。

## 7. 配置与错误处理

- 新模型文件路径加入验证配置，路径继续相对项目根目录解析。
- 缺失模型、缺失网络、占位运行参数、负载不匹配、状态维度错误、控制器状态不完整、非有限输入和日志缺失均显式报错。
- 固定负载仍为 `2.304 Ω`；首版不增加动态负载输入。
- 求解器沿用 `ode23t`、现有容差和最大步长配置；不新增另一套默认值。
- Simulink 日志输出必须覆盖请求的整个评估区间；插值或覆盖失败不得外推。

## 8. 测试与验收

自动化测试至少覆盖：

1. 构建器生成目标 `.slx`，关键块、端口、独立 PFM 和日志块存在且连线来源正确。
2. Neural ODE S-Function 对常导数测试网络给出解析一致的连续状态轨迹。
3. 未注入模型和错误初始状态被明确拒绝。
4. 冷启动双支路模型可运行，返回有限且时间覆盖完整的所有日志。
5. 预热初始化能恢复完整控制器状态，并保证两套控制器不是共享状态。
6. Neural PFM 的反馈来源是 Neural ODE 的 `v_o`，不存在从参考 `v_o` 到 Neural PFM 的误接线。
7. 现有 MATLAB 和 Python 测试继续通过。

完成后打开生成模型，执行 `FitSystem`，导出一张实际 Simulink 框图 PNG 供人工确认；截图不是测试替代品。

## 9. 非目标与限制

- 首版不做代码生成、实时硬件部署或 Simulink Coder 兼容承诺。
- 首版不把 checkpoint 或模型产物提交到 Git。
- 首版不替换现有正式误差报告流程，也不声称真实模型已经达到精度门限。
- 首版不新增动态负载、在线学习或网络权重更新。
- Level-2 MATLAB S-Function 依赖 MATLAB 执行环境，目标是离线仿真验证。
