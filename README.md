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
四路门极信号和归一化 Vin，输出四个归一化状态导数。归一化尺度计算、数据读取、
训练循环与 ODE 求解器尚未实现。配置文件由调用方读取后，将其中的 `model` mapping
传给 `llc_dynamics.core.build_neural_ode`；核心包不负责解析 YAML。

## 训练检查点

`configs/base.yaml` 已定义检查点目录、Top-K 指标与保留数量、文件名模板以及
`last.ckpt` 开关。`checkpoint.directory` 始终由调用方提供的项目根目录解析，不依赖
启动命令所在目录。Python 侧通过 `llc_dynamics.support.CheckpointConfig` 严格校验该
配置，再由 `CheckpointManager` 保存和恢复状态；当前尚无训练入口，未来训练循环负责
在合适的 epoch 调用它，也负责决定是否自动恢复。

每次保存包含模型、优化器、可选学习率调度器、epoch、global step、全部指标、运行配置
以及 Python、PyTorch 和已初始化 CUDA 设备的随机数状态。最佳文件按
`test_mean_score` 保留 5 个；同分时优先较新的 epoch/global step。`last.ckpt` 独立更新，
`index.json` 记录 Top-K 排名并绑定监控指标、方向、数量和文件名策略。文件通过同目录临时
文件与可回滚替换统一发布；配置缺失、指标非有限、optimizer/scheduler 类型或状态不兼容、
模型/归一化契约变化都会显式报错。

恢复时使用 `CheckpointManager.load(path, ..., map_location=...)` 指定文件，或使用
`load_last(...)` 恢复 `last.ckpt`。加载默认同时恢复随机数状态，可通过
`restore_rng=False` 关闭。检查点采用 PyTorch 的 pickle 序列化，只能加载由可信项目运行
产生的文件。工作区 snapshot 和训练进程启动时的自动 resume 尚未实现；若将
`save_last_snapshot` 配为 `true`，配置解析会直接报错。

MATLAB 目录已包含 LLC Simulink 模型、开环原始轨迹入口和闭环 PFM 数据生成入口，使用方法见
[`matlab/README.md`](matlab/README.md)。Python 训练功能仍未实现。
.gitkeep 仅用于让 Git 保留其余空目录。

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
