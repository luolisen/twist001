# 6 轴机械臂仿真与强化学习

这是一个面向学习、验证和二次开发的 6 自由度机械臂项目。仓库同时提供机械结构文件和 MuJoCo 仿真代码，可用于机械臂建模、正/逆运动学研究，以及基于强化学习的末端位置控制实验。

> 当前代码主要用于仿真研究，尚未包含真实电机、驱动器或传感器接口。请勿在未经限位、急停和低速测试的情况下直接连接实体机械臂。

## 功能

- 6 个旋转关节（J1～J6）的机械臂模型
- URDF 模型，可用于 ROS 或其他机器人仿真工具
- MuJoCo MJCF 模型，包含关节位置控制器、地面和可视化场景
- Gymnasium `ArmEnv` 强化学习环境
- Stable-Baselines3 PPO 训练和测试示例
- SolidWorks、STEP、STL 和材料清单等设计资料

## 仓库结构

```text
.
├── 3D/
│   ├── sw/                  # SolidWorks 零件与装配体
│   └── 加工/                 # CNC 和 3D 打印用 STEP 文件
├── RL/
│   └── src/
│       ├── arm.py           # ArmEnv、PPO 训练与测试
│       ├── demo.py          # MuJoCo 可视化演示
│       └── model/
│           ├── mjcf/arm_mjcf.xml
│           ├── urdf/arm.urdf
│           ├── meshes/       # URDF 使用的 STL 网格
│           └── img/          # 仿真场景纹理
└── 机械臂材料清单.xlsx
```

## 环境要求

- Python 3.9 或更高版本
- MuJoCo（Python 包 `mujoco`）
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy、SciPy

建议在 Linux（或 WSL2）中运行。项目使用 `SubprocVecEnv` 的 `fork` 启动方式进行并行训练；原生 Windows 不支持 `fork`，如需在 Windows 运行训练，请将 `RL/src/arm.py` 中的 `start_method="fork"` 改为适合本机的启动方式（例如 `spawn`），并根据 Stable-Baselines3 的多进程要求调整入口保护。MuJoCo 的被动窗口也需要可用的桌面环境。

## 安装

在仓库根目录执行：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install numpy scipy mujoco gymnasium stable-baselines3 torch
```

如果使用 NVIDIA GPU，请按照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装与 CUDA 匹配的版本。程序会在检测到 CUDA 时自动使用 GPU，否则回退到 CPU。

## 运行 MuJoCo 演示

代码中的模型路径是相对路径，因此请先进入 `RL/src`：

```bash
cd RL/src
python demo.py
```

演示会打开 MuJoCo 被动查看器并运行约 5 分钟。`demo.py` 默认不发送关节控制信号；如需测试随机控制，可取消代码中的 `d.ctrl[:]` 注释。

## PPO 训练与测试

`arm.py` 同时包含环境、训练函数和测试函数。文件底部通过 `TRAIN_MODE` 选择模式：

```python
TRAIN_MODE = True   # 训练
# TRAIN_MODE = False  # 测试已保存模型
```

然后在 `RL/src` 目录运行：

```bash
python arm.py
```

运行前建议先修改文件底部的 `MODEL_PATH`（例如设为 `"target"`），并按机器性能调整并行环境数。示例中的默认配置为 64 个并行环境、100,000,000 步；首次试跑建议把 `n_envs` 和 `total_timesteps` 调小，例如：

```python
train_ppo(
    n_envs=4,
    total_timesteps=100_000,
    model_save_path="target",
    visualize=False,
)
```

训练过程中按 `Ctrl+C` 中断时，程序会尝试保存带 `_interrupted` 后缀的检查点。恢复训练时设置：

```python
resume_from = "target_interrupted"
```

测试已保存模型时，将 `TRAIN_MODE` 设为 `False` 并配置模型路径：

```python
test_ppo(model_path="target", total_episodes=15)
```

训练日志默认写入 `RL/src/tensorboard/target/`，可在该目录的上一级运行：

```bash
tensorboard --logdir RL/src/tensorboard
```

### 环境接口

| 项目 | 说明 |
| --- | --- |
| 动作空间 | 6 维 `Box(-1, 1)`，映射到六个关节的目标角度 |
| 观测空间 | 9 维：6 个关节角度 + 末端目标位置 `(x, y, z)` |
| 初始姿态 | 六个关节角度均为 0 |
| 工作空间 | `x∈[-0.32, 0.38]`、`y∈[-0.32, 0.38]`、`z∈[0.09, 0.13]`（米） |
| 成功条件 | 末端与目标距离小于 0.005 米 |
| 单回合上限 | 约 20 秒，超时会终止回合 |

奖励综合考虑目标距离、末端运动的直线性、姿态（末端朝下）、动作平滑度、关节限位和碰撞等因素。具体实现请参阅 [`RL/src/arm.py`](RL/src/arm.py)。

## 模型文件说明

- `RL/src/model/mjcf/arm_mjcf.xml`：MuJoCo 仿真入口，引用 `../meshes/` 下的 STL 网格。
- `RL/src/model/urdf/arm.urdf`：通用机器人描述文件，引用同目录层级下的 `../meshes/` 网格。
- `3D/sw/`：SolidWorks 原始工程文件，编辑这些文件通常需要安装对应版本的 SolidWorks。
- `3D/加工/`：可交给 CNC 或 3D 打印流程的 STEP 文件。加工前请自行核对单位、公差、材料和装配关系。
- `机械臂材料清单.xlsx`：当前版本的零件和采购信息记录。

## 开发建议

1. 修改 MJCF 或 URDF 后，先运行 `demo.py` 检查关节方向、模型比例和碰撞体。
2. 调整奖励函数或工作空间后，再用少量步数进行 PPO 冒烟测试。
3. 训练结果、TensorBoard 日志和模型检查点建议放在单独目录，并加入 `.gitignore`，不要提交大体积实验产物。
4. 如果将模型部署到真实硬件，请重新标定零位、关节限位、减速比和控制周期，并增加硬件急停与碰撞保护。

## 许可证

本仓库当前未附带正式许可证。公开发布前，请在根目录添加 `LICENSE` 文件（例如 MIT 或 Apache-2.0），并确认 SolidWorks、第三方模型、纹理及材料清单中的内容均具备再分发权限。

## 贡献与反馈

欢迎提交 Issue 或 Pull Request 来报告问题、改进模型、补充控制器和训练结果。提交问题时请附上操作系统、Python 版本、依赖版本，以及可复现问题的最小步骤。
