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

## 初始状态

Python 最低版本为 3.11。requirements.txt 当前不声明依赖，实现功能时按需添加。
pyproject.toml 仅保留打包元信息，不注册命令入口。
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
