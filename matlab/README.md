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
[files, summary] = generate_llc_closed_loop_dataset();
```

工况唯一来源是 `configs/matlab/closed_loop_condition_plan.json`。运行
`build_llc_closed_loop_generation_configs` 可确定性生成训练、内插、负载外推、目标电压外推和高压外推
五份配置；脚本校验每组数量、组内唯一性及跨组隔离。默认训练配置仍为
`configs/matlab/closed_loop_data_generation.json`，包含 `6 × 4 × 6 = 144` 条四状态零初值启动轨迹。
全部五组共 182 条，每条工况内三个参数保持恒定，并共用 1 ms 参考爬升和控制器参数。闭环 PI 不配置
频率上下限，也不对频率指令做饱和裁剪；频率变化率、死区和最小脉宽仍由配置给定。

每个工况统一仿真并保存到 5 ms。默认在 2.0–5.0 ms 之间每隔 0.1 ms 检查一次；稳态结果只作为
标签，不缩短固定网格。仿真先保留 `ode23t` 到 5 ms 的完整变步长原始输出，随后按
`sampling.reference_frequency_hz` 和 `sampling.intervals_per_cycle` 构造固定网格。状态在网格节点上线性
插值；实际门极、频率指令和其余控制量在每个网格区间中点按前值采样。MAT 文件同时保存固定网格
数据、完整原始求解器数据、原始仿真终止时刻和采样审计信息。`initial_gate_values`、
`gate_event_times_s` 和 `gate_values_after` 从实际门极日志提取，保留未吸附到固定网格的准确变化时刻；
中点门极值继续保留用于检查和兼容读取。

稳态检查使用当前检查点之前 0.4 ms 内完整的实际门极周期。`regulated` 要求所有输出电压周期均值
位于 Vref 的 ±2% 内；`dynamics_stable` 同时检查四个状态的周期均值和峰峰值纹波是否停止漂移。
两者都成立才记为 `settled`。`settled_at_first_checkpoint` 表示配置中的首个检查点通过，默认首个检查点为
2.0 ms。具体参考尺度和容差均在 `steady_state_assessment` 中配置。

每个候选周期还必须包含“正对角导通—死区—负对角导通—死区”的完整四路门极序列；缺相、漏周期
或截断序列不会被判为稳态。周期识别直接使用实际门极跳变，不使用预设频率范围。频率必须保持有限
且为正，死区和最小脉宽必须能实际生成；这些属于波形结构有效性检查，并不是数据筛选用的频率边界。
门极时序、初始状态和终止时刻的数值校验容差位于 `validation_tolerances`。

达到 5 ms 仍未确认稳态的数值有效轨迹也会保存，并带有 `settled=false` 标签。单个工况发生
仿真、校验或写文件错误时，生成器发出警告并继续处理后续工况。`files` 与配置工况一一对应，失败项
为空字符串；`summary.entries` 记录每项的成功状态、输出路径、稳态结果或错误标识与消息。

参考频率为 110 kHz、每个参考周期 100 个区间时，5 ms 轨迹包含 55,001 个状态点和 55,000 个
区间信号，固定采样间隔约为 90.91 ns。

可按集合分批运行并续接已有有效输出：

```matlab
results = generate_llc_closed_loop_suite("train", true);
results = generate_llc_closed_loop_suite( ...
    ["interpolation", "load_extrapolation", ...
     "reference_extrapolation", "high_voltage_extrapolation"], true);
```

生成器保留所有成功仿真且结构有效的轨迹，不用 `regulated` 或 `settled` 标签拒收数据。训练前应结合
这些标签、状态和器件应力进行独立的数据质量检查；当前策略不构成实际硬件频率范围或 ZVS 安全认证。

## 通用数据检查

生成数据后可运行与调制策略无关的批量检查：

```matlab
addpath('matlab');
setup_paths;
[report, summary, stateMetrics, controlMetrics] = ...
    llc_validate_generated_dataset();
```

入口默认读取 `configs/matlab/dataset_validation.json`，扫描配置中的轨迹目录，并把报告写到
`artifacts/data_validation`。也可以把第一个参数指定为单个 MAT 文件、MAT 文件字符串数组或其他目录，
第三个参数指定报告目录。

检查分为两类。结构检查验证字段和变量顺序、时间轴、有限数值、零初态、固定网格对齐、二值门极、
同一桥臂互锁以及原始数据覆盖范围；失败项通过稳定的问题代码记录。统计检查输出各状态的全程峰值和
尾部范围、各控制量的范围及末值，并根据配置计算尾部输出均值、峰峰值和参考误差。输出误差超限只写入
`output_tail_within_tolerance=false`，不会把结构有效的数据删除或判为损坏。

通用检查不假定“正对角—死区—负对角—死区”等特定开关顺序，因此可以复用于 PFM、PWM 和移相数据。
具体调制策略的周期完整性、死区长度、ZVS 及器件额定应力需要由独立检查器补充。

报告包含四个文件：轨迹汇总 CSV、状态统计 CSV、控制量统计 CSV，以及保存完整配置和上述表格的 MAT
文件。默认文件名前缀和检查阈值都在 `dataset_validation.json` 中配置。

## 独立的 Vin/Vref 正弦激励数据

正弦激励流程与上述固定工况闭环流程使用同一个
`llc_full_bridge_closed_loop.slx`，但使用独立配置、入口和输出目录。旧入口仍生成每条轨迹内 Vin、Vref
固定的数据；新入口只负责 Vin/Vref 连续正弦激励，两者不互相调用。

首轮筛选配置位于
`configs/matlab/sinusoidal_excitation/screening_data_generation.json`，负载固定为 2.304 Ω，包含 Vin
单独激励和 Vref 单独激励各三条轨迹。运行：

```matlab
addpath('matlab');
setup_paths;
[files, summary] = generate_llc_sinusoidal_excitation_dataset();
```

输出写入 `data/raw/llc_sinusoidal_excitation/screening`。Vin 和 Vref 在 5 ms 开始，经 2 ms 半余弦包络
进入正弦激励；每条轨迹保存到 57 ms。新配置为延长后的动态轨迹显式设置 60–250 kHz 频率范围，并在
饱和方向冻结积分。该设置只存在于正弦激励配置，旧固定工况配置仍采用原有无频率边界策略。
网络输入所需的 Vin 与四路实际门极均保存到固定网格，Vref、
频率指令及原始求解器输出作为控制与审计信号同时保存。正弦数据不使用固定参考值的稳态判据。

## 连续随机分段激励正式数据

正式配置位于
`configs/matlab/sinusoidal_excitation/continuous_random_data_generation.json`。它保留上述筛选配置，仍由独立的
Vin/Vref 外部激励入口生成数据：

```matlab
addpath('matlab');
setup_paths;
config = fullfile(pwd, 'configs', 'matlab', 'sinusoidal_excitation', ...
    'continuous_random_data_generation.json');
[files, summary] = generate_llc_sinusoidal_excitation_dataset([], config, true);
```

配置包含 120 条 45 ms 轨迹，其中前 5 ms 用于启动，随后包含 2 个 20 ms 激励段。每段开始时重新抽取
Vin 和/或 Vref 的中心值、幅值与频率，并在 2 ms 内使用三次平滑步函数过渡；频率经积分形成连续相位，
因此激励值及一阶导数在更新点连续。负载固定为 2.304 Ω。

Vin 的中心值范围为 380–420 V、峰值幅度为 10–20 V；Vref 的中心值范围为 45.6–50.4 V、峰值幅度为
1.2–2.4 V；两者的频率均从 100、300、1000 Hz 中独立抽取。每条轨迹保存独立随机种子与全部实际分段参数。
数据按整条轨迹划分为 80 条训练、20 条验证和 20 条测试，包含 30 条仅 Vin、30 条仅 Vref 和 60 条双变量激励。

固定网格采样率为 11 MHz。每条轨迹包含
`0.045 * 110000 * 100 + 1 = 495001` 个状态时刻，120 条共 59400120 个状态时刻。网络输入契约仍为
4 个状态、4 路实际门极和 Vin；Vref 只保存为控制与审计信号。输出按数据划分和激励类型写入
`data/raw/llc_sinusoidal_excitation/continuous_random/<split>/<excitation_group>`，其中 `split` 为
`train`、`validation` 或 `test`，`excitation_group` 为 `vin_only`、`reference_only` 或 `both`。文件使用 MAT v7。
MATLAB 的 HDF5/v7.3 驱动无法直接在本项目的
WSL UNC 路径创建或重新打开文件，而单条轨迹小于 MAT v7 的变量大小限制。完整批量生成耗时较长，不属于自动测试；
测试使用相同正式配置架构的缩短轨迹验证端到端生成链路。
