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
