# 机械设计与 REV6 仿真

当前独立仿真入口是 [REV6 双相机仿真](RL/README.md)，包括完整夹爪、腕部相机、前置相机以及任务场景。旧 PPO 脚本及独立仿真目录中的旧 URDF 已从当前版本替换；历史内容仍可从 Git 历史查看。

- [仿真启动、依赖及相机合同](RL/README.md)
- [模型来源与文件指纹](RL/provenance.json)
- [三模型阶段监督与技能研究](three_model_training/README.md)
- [历史机械设计资料](3D/) 与 [材料清单](机械臂材料清单.xlsx)：本次未更新，不代表已与 REV6 一致。

相机是实拍对齐候选，完整光学校准与实机运动验证未完成。模型加载及渲染验证不代表任务成功率或碰撞安全验收。

2026-10-10研究记录已补入：已知场景拿起保持及局部重力前馈跟踪通过；完整Transport/Place、公开自主交接和实机仍未通过。源码与精简证据见[研究归档](three_model_training/experiments/skill_supervision_20261010/README.md)。
