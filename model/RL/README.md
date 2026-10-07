# REV6 双相机仿真

当前版本完整替换旧独立仿真目录，包含 REV6 机械臂、联动双指夹爪、腕部相机及对齐后的前置相机。
默认 `src/model/mjcf/arm_mjcf.xml` 是相机对齐场景；`task_scene.xml` 是同相机版本的方块与放置目标任务场景。

| 相机 | 视角 | 原始渲染 | 策略图像 | 内容与补边 |
| --- | --- | --- | --- | --- |
| front_rgb | 48° | 640×360 | 256×256 | 256×144，上下各 56 |
| wrist_camera | 45.2° | 640×360 | 512×512 | 512×288，上下各 112 |

保持完整 16:9 视野，不用中心裁剪。前置相机世界位置为 (0.3665, 0.029, 0.355) m；机械臂绕 (0.08, 0.08, 0) 旋转 −90°。相机姿态来自实拍人工对齐，尚未完成逐设备内参、畸变及全运动可见性验收。

## 使用

在仓库根目录安装依赖并运行：

```sh
python -m pip install -r model/RL/requirements.txt
python model/RL/src/demo.py
python model/RL/src/demo.py --scene task
python model/RL/src/demo.py --check /tmp/rev6-camera-check
```

macOS 打开交互窗口使用环境中的 `mjpython model/RL/src/demo.py`。离屏检查使用普通 Python。
仿真启动姿态 J1–J6 为 [0, −0.05, 0.1, 0, 0, 0] rad；控制接口为六个关节目标加夹爪开度（米，0–0.08）。这不是实机上电零位。

## 来源和验证边界

基体来源 `ARM_REV6_CAMERA45P2_MUJOCO`，SHA-256 为 `49cc167d19aba3cf02702e98f1f95e64fabc3dab05aa267ec67d476d4028c7eb`。
相机对齐场景源 SHA-256 为 `986230eea6327b3e1f6a4b470eb44d1582c53c0ac463c3670da4547319da9bc8`；任务场景源为 `02defa53b8d934771f32aed1d0b767bd1a6a820715d5fc0a7eaffa46189cf9ce`。
打包仅修改资源相对路径及 XML 序列化；几何、相机和动力学没有改动，发布文件的新 hash 见 `provenance.json`。

旧六轴 PPO 的 arm.py/test.py 与旧 URDF 副本不纳入本目录；它们不适配当前夹爪、相机和任务接口。三模型历史训练记录保留在 `model/three_model_training/`。
本目录不含数据集、训练权重或自动训练入口。模型加载、图像渲染及短步数烟测不等于抓取成功、碰撞验收、策略可靠性或实机验证。默认对齐场景的桌板是视觉几何；任务场景的桌板有碰撞几何。未合入静态碰撞修复候选。
