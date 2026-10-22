# MATLAB 数据生成

本目录保留 LLC 全桥 Simulink 电路，并提供最小的原始轨迹生成流程。硬件参数与数据生成参数分别位于
`configs/matlab/hardware.json` 和 `configs/matlab/data_generation.json`，MATLAB 代码不保存工况或数值求解器常量。

在 MATLAB 中切换到项目根目录后运行：

```matlab
addpath('matlab');
setup_paths;
files = generate_llc_dataset();
```

每个工况生成一个 MAT 文件。顶层字段包括状态与门极各自的原生求解器时间轴、四维状态
`i_Lr/v_Cr/i_Lm/v_o`、四路实际门极、工况、变量名与单位，以及本次运行的完整生效配置。

当前入口仅支持一条轨迹内恒定的输入电压、负载电阻和开关频率。完整的零初态启动过程会被保存，
不会自动预热、截断、插值、重采样、划分数据集或生成训练窗口。

## 闭环 PFM 数据生成

`llc_full_bridge_closed_loop.slx` 在同一功率电路上加入输出电压反馈。离散 PI 根据滤波后的输出电压
计算频率指令，事件驱动 PFM 保持相位连续，并在全桥对角开关换相时插入固定死区。运行：

```matlab
addpath('matlab');
setup_paths;
files = generate_llc_closed_loop_dataset();
```

参数位于 `configs/matlab/closed_loop_data_generation.json`。每个 `conditions` 元素生成一条独立轨迹，
可分别设置输入电压、负载、参考电压和仿真时长。除状态与实际门极外，MAT 文件还保存频率指令、
滤波输出电压、参考电压、控制误差和 PI 积分状态。

默认 PI 参数已在 400 V、2.304 Ω、48 V 参考的仿真工况下验证。改变硬件参数或扩大输入与负载范围前，
需要重新检查闭环稳定性、频率饱和、软启动和器件应力。
