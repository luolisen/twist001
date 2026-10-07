# REV6 MuJoCo 资源

MuJoCo 使用的 `mjcf/`、`meshes/`、`img/` 资源与 `model/RL/src/model/` 完全一致，包含最新前置 48° / 腕部 45.2° 相机。
来源、相机合同、启动方法与验证边界见仓库 `model/RL/README.md`。

`meshes/j6_Link.STL` 仅为旧 URDF 保留，不被 REV6 MJCF 引用。
`urdf/` 保留旧 ROS 控制/标定依赖所需的历史六轴描述，未将其冒充为 REV6 夹爪及相机 URDF；它不参与此 MuJoCo 模型加载。ROS 控制链与该历史 URDF 的一致性尚未验收。
现有 mujoco_sim 节点仍提供六关节话题，不发布相机图像或夹爪指令。本次仅更新模型资源与初始关节姿态；未执行 ROS 2 构建或整链路验证。
