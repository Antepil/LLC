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
MATLAB 目录已包含 LLC Simulink 模型和最小原始轨迹生成入口，使用方法见
[`matlab/README.md`](matlab/README.md)。Python 训练功能仍未实现。
.gitkeep 仅用于让 Git 保留其余空目录。
