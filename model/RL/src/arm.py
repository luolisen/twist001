"""六轴机械臂的 MuJoCo 强化学习环境与 PPO 训练脚本。

运行本文件前请将当前工作目录切换到 ``RL/src``，因为模型文件使用了
相对于当前目录的路径（例如 ``./model/mjcf/arm_mjcf.xml``）。
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
import signal
from typing import Optional
from scipy.spatial.transform import Rotation as R

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

import os

def write_flag_file(flag_filename="rl_visu_flag"):
    """在临时目录创建可视化标记，避免并行环境重复打开窗口。"""
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    """检查可视化标记是否已经存在。"""
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_visu_flag"):
    """删除可视化标记，通常在主程序启动前调用。"""
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

class ArmEnv(gym.Env):
    """用于训练末端位置控制策略的 Gymnasium 环境。

    动作是 6 个关节的归一化目标值，观测由当前 6 个关节角度和随机生成的
    末端目标位置组成。每个回合从固定的 home 姿态开始。
    """

    def __init__(self, visualize: bool = False):
        super(ArmEnv, self).__init__()
        # SubprocVecEnv 会创建多个环境。临时标记保证最多只有一个环境启动查看器。
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        # MJCF 中还引用了 mesh 和纹理，因此必须从 RL/src 目录启动脚本。
        self.model = mujoco.MjModel.from_xml_path('./model/mjcf/arm_mjcf.xml')
        self.data = mujoco.MjData(self.model)
        
        if self.visualize:
            # 使用被动查看器：仿真由 step() 驱动，查看器只负责显示当前状态。
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])
        
        # 末端 body 的 ID 用于读取末端位置和姿态。
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.initial_ee_pos = np.zeros(3, dtype=np.float32) 
        self.home_joint_pos = np.array([  
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ], dtype=np.float32)
        
        self.goal_size = 0.03  # 查看器中目标球的半径（米）
        
        # 约束工作空间
        self.workspace = {
            'x': (-0.32, 0.38),
            'y': (-0.32, 0.38),
            'z': (0.09, 0.13)
        }
        
        # 动作空间与观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # 6轴关节角度、目标位置
        self.obs_size = 6 + 3 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        
        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.prev_action = np.zeros(6, dtype=np.float32)  # 用于动作平滑惩罚
        self.goal_threshold = 0.005  # 末端距离目标小于 5 mm 时视为成功

    def _get_valid_goal(self) -> np.ndarray:
        """在限定工作空间内采样目标点，并过滤过近或过远的目标。"""
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            if np.linalg.norm(goal - self.initial_ee_pos) > 0.1 and np.linalg.norm(goal - self.initial_ee_pos) < 0.32:
                return goal.astype(np.float32)

    def _render_scene(self) -> None:
        """在 MuJoCo 用户场景中绘制蓝色目标球。"""
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染目标点（蓝色）
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=goal_rgba
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """重置仿真状态并返回新回合的初始观测。"""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # 清除动力学状态，再将六个关节设置到固定的 home 姿态。
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.home_joint_pos
        mujoco.mj_forward(self.model, self.data)
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.start_ee_pos = self.initial_ee_pos.copy()
        
        # 根据初始末端位置生成本回合目标，并刷新可视化目标球。
        self.goal = self._get_valid_goal()
        if self.visualize:
            self._render_scene()        
        
        obs = self._get_observation()
        self.start_t = time.time()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        """拼接观测：关节角度（6 维）和目标坐标（3 维）。"""
        joint_pos = self.data.qpos[:6].copy().astype(np.float32)
        # ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        # ee_quat = self.data.body(self.end_effector_id).xquat.copy().astype(np.float32)
        return np.concatenate([joint_pos, self.goal])


    def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        """计算单步奖励，并返回奖励、末端到目标距离和姿态误差。

        奖励鼓励末端接近目标并沿起点到目标的直线运动，同时惩罚碰撞、姿态
        偏差、动作突变、关节越界和偏离直线后的回退行为。
        """
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)
    
        # 非线性距离奖励（保持不变）
        if dist_to_goal < self.goal_threshold:
            distance_reward = 100.0
        elif dist_to_goal < 2*self.goal_threshold:
            distance_reward = 50.0
        elif dist_to_goal < 3*self.goal_threshold:
            distance_reward = 10.0
        else:
            distance_reward = 1.0 / (1.0 + dist_to_goal)

        # 计算起点到目标的向量及相关参数
        start_to_goal = self.goal - self.start_ee_pos
        start_to_goal_norm = np.linalg.norm(start_to_goal)
        linearity_reward = 0.0
        deviation_penalty = 0.0
        
        if start_to_goal_norm >= 1e-6:  # 起点和目标不重合时才计算直线相关奖励/惩罚
            # 计算当前位置到起点的向量
            start_to_current = ee_pos - self.start_ee_pos
            # 计算当前位置在“起点→目标”直线上的投影比例（限制在0~1，避免超出目标后惩罚）
            projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm **2)
            projection_ratio = np.clip(projection_ratio, 0.0, 1.0)
            # 计算直线上的投影点，得到当前位置偏离直线的垂直距离
            projected_point = self.start_ee_pos + projection_ratio * start_to_goal
            linearity_error = np.linalg.norm(ee_pos - projected_point)  # 偏离直线的距离
            
            # 1. 直线接近奖励：离直线越近，奖励越高（非线性递增）
            linearity_reward = 3.0 / (1.0 + linearity_error)  # 系数8.0可根据重要性调整
            
            # 2. 远离趋势惩罚：检测“先靠近后远离”的行为
            # 初始化或更新历史最小偏离距离（跟踪最近点）
            if not hasattr(self, 'min_linearity_error'):
                self.min_linearity_error = np.inf  # 首次运行初始化
            if linearity_error < self.min_linearity_error:
                self.min_linearity_error = linearity_error  # 更近时更新最小值，无惩罚
            else:
                # 比最近点更远时，惩罚远离的程度（距离差越大，惩罚越重）
                deviation_penalty = 1.0 * (linearity_error - self.min_linearity_error)  # 系数3.0可调整

        # 姿态约束：保持末端朝下（保持不变）
        target_orient = np.array([0, 0, -1])
        ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)
        dot_product = np.dot(ee_orient_norm, target_orient)
        angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
        orientation_penalty = 0.3 * angle_error
        
        # 动作相关惩罚：相邻动作变化越大，轨迹越不平滑。
        action_diff = action - self.prev_action
        smooth_penalty = 0.1 * np.linalg.norm(action_diff)
        # 保留动作幅值项，便于后续调参；当前总奖励未使用该项。
        action_magnitude_penalty = 0.05 * np.linalg.norm(action)

        # 碰撞惩罚（保持不变）
        contact_reward = 1.0 * self.data.ncon
        
        # 关节角度限制惩罚（保持不变）
        joint_penalty = 0.0
        for i in range(6):
            min_angle, max_angle = self.model.jnt_range[:6][i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
        
        # 总奖励：整合新的直线奖励和远离惩罚
        total_reward = (distance_reward 
                    + linearity_reward  
                    - contact_reward 
                    - smooth_penalty 
                    - orientation_penalty 
                    - joint_penalty 
                    - deviation_penalty)  
        
        # 更新上一步动作
        self.prev_action = action.copy()
        
        return total_reward, dist_to_goal, angle_error

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        """执行一个仿真步并返回 Gymnasium 标准的五元组。"""
        # PPO 输出范围为 [-1, 1]，将其线性映射到 MJCF 中每个关节的角度范围。
        joint_ranges = self.model.jnt_range[:6]
        scaled_action = np.zeros(6, dtype=np.float32)
        for i in range(6):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 位置执行器接收的是目标关节角度，MuJoCo 负责推进动力学仿真。
        self.data.ctrl[:6] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 读取末端状态；MuJoCo 四元数顺序转换为欧拉角后用于姿态惩罚。
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        rot = R.from_quat(ee_quat)
        ee_quat_euler_rad = rot.as_euler('xyz')
        reward, dist_to_goal,_ = self._calc_reward(ee_pos, ee_quat_euler_rad, self.data.qpos[:6], action)
        terminated = False
        # 当前 info 字段尚未逐接触判断碰撞；奖励中的碰撞项直接使用 data.ncon。
        collision = False
        
        # 目标达成即终止回合；否则超过 20 秒也会因超时终止。
        if dist_to_goal < self.goal_threshold:
            terminated = True
        # print(f"[奖励] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        if not terminated:
            if time.time() - self.start_t > 20.0:
                reward -= 10.0
                # print(f"[超时] 时间过长，奖励减半")
                terminated = True

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01) 
        
        obs = self._get_observation()
        info = {
            'is_success': terminated and (dist_to_goal < self.goal_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }
        
        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        """兼容旧版 Gym API 的随机种子接口。"""
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        """关闭查看器并释放环境资源。"""
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 40_000_000,  # 本次训练的新增步数
    model_save_path: str = "target",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:
    """使用 PPO 训练末端到达策略。

    Args:
        n_envs: 并行采样的环境数量。
        total_timesteps: 本次新增的训练步数。
        model_save_path: 正常结束或中断时的模型保存路径（SB3 会追加 .zip）。
        visualize: 是否在环境中启用 MuJoCo 查看器；并行训练通常应设为 False。
        resume_from: 已有模型路径；为 None 时创建新的 PPO 模型。
    """
    ENV_KWARGS = {'visualize': visualize}
    env = None
    model = None
    interrupted = False
    old_signal_handlers = {}

    # SB3 不会自动保存被中断的 learn() 调用。将系统信号转换为
    # KeyboardInterrupt，才能在清理并行环境前保存内存中的模型。
    def _handle_training_signal(signum, frame):
        raise KeyboardInterrupt

    def _interrupt_model_path(path: str) -> str:
        """生成中断模型的固定保存路径，避免重复追加 ``.zip``。"""
        path = os.fspath(path)
        if path.endswith(".zip"):
            path = path[:-4]
        return f"{path}_interrupted"

    try:
        # 使用 SubprocVecEnv 并行采样，提高机械臂仿真数据的收集速度。
        env = make_vec_env(
            env_id=lambda: ArmEnv(**ENV_KWARGS),
            n_envs=n_envs,
            seed=42,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"}
        )

        if resume_from is not None:
            # 加载模型时重新绑定当前环境，以便继续采样和训练。
            model = PPO.load(resume_from, env=env)
        else:
            # 策略网络分别为 actor 和 critic 配置两层全连接层。
            POLICY_KWARGS = dict(
                activation_fn=nn.ReLU,
                net_arch=[dict(pi=[256, 128], vf=[256, 128])]
            )
            model = PPO(
                policy="MlpPolicy",
                env=env,
                policy_kwargs=POLICY_KWARGS,
                verbose=1,
                n_steps=2048,
                batch_size=2048,
                n_epochs=10,
                gamma=0.99,
                learning_rate=2e-4,
                device="cuda" if torch.cuda.is_available() else "cpu",
                tensorboard_log="./tensorboard/target/"
            )

        # signal.signal 只能在主线程中调用；线程内调用训练时忽略该设置即可。
        try:
            old_signal_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
            old_signal_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, _handle_training_signal)
            signal.signal(signal.SIGTERM, _handle_training_signal)
        except ValueError:
            old_signal_handlers.clear()

        print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
        try:
            # progress_bar=True 会显示 Stable-Baselines3 的训练进度。
            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True
            )
        except KeyboardInterrupt:
            interrupted = True
            # 中断模型使用稳定的后缀命名，便于后续通过 resume_from 恢复训练。
            interrupt_path = _interrupt_model_path(model_save_path)
            try:
                model.save(interrupt_path)
                print(f"检测到中断，当前训练模型已保存至: {interrupt_path}")
            except Exception as save_error:
                # 保存失败时打印错误，但不要吞掉训练中断本身。
                print(f"检测到中断，但模型保存失败: {save_error}")
    finally:
        # 无论训练成功还是异常退出，都恢复原信号处理器并关闭子环境。
        for signum, handler in old_signal_handlers.items():
            signal.signal(signum, handler)
        if env is not None:
            env.close()

    if not interrupted:
        model.save(model_save_path)
        print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "target",
    total_episodes: int = 5,
) -> None:
    """加载训练好的 PPO 模型，在可视化环境中评估若干回合。"""
    env = ArmEnv(visualize=True)
    model = PPO.load(model_path, env=env)
    
    
    success_count = 0
    print(f"测试轮数: {total_episodes}")
    
    for ep in range(total_episodes):
        # 每个回合从 home 姿态开始，并让策略确定性地选择动作。
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")
    
    success_rate = (success_count / total_episodes) * 100
    print(f"总成功率: {success_rate:.1f}%")
    
    env.close()


if __name__ == "__main__":
    # 清理上一次进程留下的可视化标记，避免本次运行被误判为并行环境。
    delete_flag_file()
    TRAIN_MODE = True  # True：训练；False：加载模型并测试
    MODEL_PATH = "./model_set"  # 模型保存/加载路径（SB3 会使用 .zip 文件）
    RESUME_MODEL_PATH = ""  # 例如 "./model_set_interrupted"
    if TRAIN_MODE:
        train_ppo(
            n_envs=64,                # 并行环境数，按 CPU 内存和核心数调整
            total_timesteps=100_000_000,  # 训练总步数
            model_save_path=MODEL_PATH,
            visualize=True,  # 并行训练建议改为 False，避免启动查看器
            # resume_from=RESUME_MODEL_PATH  # 从已有检查点继续训练
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=15,  # 评估回合数
        )
