# LLC Neural ODE — 重建骨架

独立的最小项目，保留原项目目录结构，用于重新开始开发。
不包含旧源码、模型、实验配置、数据或实验结果。

## 目录职责

- `src/llc_dynamics/core/`：核心算法。
- `src/llc_dynamics/support/adapters/`：辅助模块及数据适配。
- `matlab/`：模型、调制、调度、仿真、导出与测试。
- `configs/`：配置及实验配置。
- `data/`：原始数据、归一化数据、集合划分与窗口。
- `artifacts/`：实验产物。
- `scripts/`：工程脚本。
- `tests/`：单元测试、集成测试与测试素材。

## Python 环境

Python 最低版本为 3.11。神经 ODE 网络使用 CUDA 版 PyTorch；推荐在独立环境中安装：

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` 固定使用 CUDA 13.0 wheel。安装后可检查 CUDA 构建和 GPU 是否可用：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

基础配置位于 `configs/base.yaml`。当前网络接收已经归一化的九维张量：四个状态、
四路门极信号和归一化 Vin，输出四个归一化状态导数。配置文件由调用方读取后，将其中的 `model` mapping
传给 `llc_dynamics.core.build_neural_ode`；核心包不负责解析 YAML。

固定负载首版通过 `NormalizationConfig` 仅在训练集上拟合四状态的非中心化 RMS 尺度；
`LLCNormalizer` 完成状态、Vin、时间及状态导数的双向换算。基础配置中的 `state_scales`
刻意保持为 `null`，拟合后必须使用 `resolved(...).as_mapping()` 生成包含四个明确尺度的
运行配置，并把它交给检查点管理器。验证集和测试集不得参与尺度拟合。

`llc_dynamics.support.load_matlab_trajectory` 读取固定 Vin 或正弦激励的 MATLAB v5/v7
轨迹，统一返回状态节点、节点 Vin 和准确门极事件表。读取时严格校验状态/门极顺序、时间轴、
有限值、二值门极、桥臂互斥、事件覆盖范围及固定负载契约。调用方必须把完整运行配置传给
加载器；负载约束只读取 `condition.load_resistance_ohm`，加载器内部没有独立默认值。

训练目标为 `NormalizedRolloutMSE`：比较数值积分器生成的 `[batch, time, 4]` 归一化
状态序列与目标序列，对除初始节点之外的所有元素取均方误差。损失模块不计算有限差分导数，
也不执行 ODE 积分。

`TrajectoryWindowDataset` 按 `configs/base.yaml` 的 `window` 配置从已经完成集合划分的完整轨迹
惰性生成定长窗口。每个窗口只用真实起始状态初始化，保留准确门极事件和节点 Vin；起点事件
决定初始门极，终点事件不进入当前窗口。`collate_rollout_windows` 堆叠定长状态和 Vin，同时将
各窗口数量不同的门极事件保留为独立张量。

`rollout_window` 和 `rollout_batch` 使用可微的固定步长 RK4 生成归一化状态序列。求解器在准确
门极事件处分段，事件前的分段始终使用旧门极；Vin 在每个 RK4 阶段按相邻状态节点线性插值。
`solver.max_step_tau` 是归一化时间中的最大子步长，所有分段都会细分到不超过该值。batch 中
不同事件表的窗口逐个积分后堆叠。

## 训练

训练入口按 `training.dataset_manifest` 核对正式的 120 条 continuous-random 轨迹，并从
`training.data_directory` 下的 `train`、`validation`、`test` 子目录递归读取 MATLAB 轨迹。
程序严格检查 80/20/20 数量、轨迹 ID、激励分组、目录归属和文件 SHA-256；三个集合按完整轨迹
隔离，状态尺度只从训练集合拟合；随后按窗口构造 DataLoader，
执行事件感知 rollout、反向传播、AdamW 参数更新、验证集评估和 Top-K/last 检查点保存。训练结束
后报告测试集指标，但测试集不参与选模。运行：

```bash
source .venv-gpu/bin/activate
python scripts/train.py --config configs/base.yaml
```

终端日志采用 Cuk 项目的阶段化布局：数据加载显示各集合进度；首个、最后一个及每隔
`training.log_every_steps` 个优化器 update 输出一次损失；每轮结束输出 train/validation MSE、
总耗时、分项耗时和 batch 数。完整的逐轮记录同时追加到检查点目录的 `training.jsonl`。

`training.resume_last: true` 时，如果检查点目录已有 `last.ckpt`，训练先恢复其中的归一化尺度，
再严格核对模型、归一化契约和完整数据身份，从下一 epoch 继续。正式训练前需先按
`matlab/README.md` 生成 continuous-random 数据；缺少任何正式轨迹都会显式报错。训练结束后
加载验证指标排名第一的检查点执行测试集评估，而不是直接评估最后一个 epoch。

数据仍在生成时，可用 `--available-snapshot` 固定启动时已经完成的正式轨迹，并跳过尚不可用的
测试集评估。阶段性训练必须使用独立的检查点目录；后续新增文件不会进入已启动的进程，且由于
数据身份不同也不能混入同一次续训。例如：

```bash
python scripts/train.py --config configs/base.yaml --available-snapshot \
  --checkpoint-directory artifacts/checkpoints/available_snapshot_YYYYMMDD_HHMMSS
```

## 训练检查点

`configs/base.yaml` 已定义检查点目录、Top-K 指标与保留数量、文件名模板以及
`last.ckpt` 开关。`checkpoint.directory` 始终由调用方提供的项目根目录解析，不依赖
启动命令所在目录。Python 侧通过 `llc_dynamics.support.CheckpointConfig` 严格校验该
配置，再由 `CheckpointManager` 保存和恢复状态；训练循环在每个 epoch 验证后调用它，并按
`training.resume_last` 决定是否自动恢复。

每次保存包含模型、优化器、可选学习率调度器、epoch、global step、全部指标、运行配置
以及 Python、PyTorch 和已初始化 CUDA 设备的随机数状态。最佳文件按验证集指标
`val_rollout_mse` 从小到大保留 5 个；同分时优先较新的 epoch/global step。测试集不参与选模。
`last.ckpt` 独立更新，
`index.json` 记录 Top-K 排名并绑定监控指标、方向、数量和文件名策略。文件通过同目录临时
文件与可回滚替换统一发布；配置缺失、指标非有限、optimizer/scheduler 类型或状态不兼容、
模型/归一化契约变化都会显式报错。

恢复时使用 `CheckpointManager.load(path, ..., map_location=...)` 指定文件，或使用
`load_last(...)` 恢复 `last.ckpt`。加载默认同时恢复随机数状态，可通过
`restore_rng=False` 关闭。检查点采用 PyTorch 的 pickle 序列化，只能加载由可信项目运行
产生的文件。工作区 snapshot 和训练进程启动时的自动 resume 尚未实现；若将
`save_last_snapshot` 配为 `true`，配置解析会直接报错。

MATLAB 目录已包含 LLC Simulink 模型、开环原始轨迹入口和闭环 PFM 数据生成入口，使用方法见
[`matlab/README.md`](matlab/README.md)。
.gitkeep 仅用于让 Git 保留其余空目录。

已训练模型可通过 `scripts/export_matlab_model.py` 导出为原生 MATLAB 网络契约，再由
`llc_run_neural_ode_validation` 执行准确门极回放、冷启动和预热交接后的独立闭环对照。配置、信号时间轴、
运行步骤与产物说明见 [`matlab/README.md`](matlab/README.md#neural-ode-仿真验证)。

仓库同时提供 `matlab/models/llc_neural_ode_comparison.slx`，可在一个 Simulink 图中并行运行参考 LLC
电路和 Neural ODE。两条支路共享 Vin/Vref，但各自闭合 PI/PFM 反馈，并直接记录状态、门极、控制量和
四状态误差。配置化的冷启动/预热交接入口、模型重建方法和两种验证后端的区别见
[`matlab/README.md`](matlab/README.md#simulink-双分支波形对照)。

### Closed-loop frequency policy

The closed-loop PI has no configured minimum/maximum frequency and no saturation
anti-windup. The configured frequency slew rate, deadtime and minimum pulse remain.
Nonpositive/nonfinite applied frequencies or impossible switching timing fail the
condition explicitly; commands are never silently clipped to a replacement bound.
Obsolete `minimum_frequency_hz` / `maximum_frequency_hz` configuration fields are rejected.
Steady-state cycle recognition uses actual bridge transitions, not an assumed frequency band.
A deadtime interval as long as an active interval is marked invalid for steady-state
assessment (including missing-cycle gaps); this does not discard the trajectory.
All successfully simulated, structurally valid trajectories are retained, including
unregulated or unsettled cases. Every reviewed condition is saved through 5 ms; steady-state
assessment runs from 2 ms through 5 ms without shortening the fixed grid. Raw applied-gate
events are exported independently of the 90.91 ns state grid. This policy does not certify
hardware ratings or ZVS.
