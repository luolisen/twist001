import time
import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# MuJoCo Null-Space Demo — PPT Video Edition
# ============================================================
#
# Stage 0:
#   初始状态停留片刻，方便视频看清楚
#
# Stage 1:
#   末端从初始位置向前、向上运动
#
# Stage 2:
#   锁定末端世界坐标 XYZ
#   同时机械臂在零空间中做大幅、平滑的构型变化
#
# 注意：
#   只固定末端 POSITION
#   不固定末端 ORIENTATION
#
# ============================================================


MODEL_PATH = "./model/mjcf/arm_mjcf.xml"


JOINT_NAMES = [
    "j1_joint",
    "j2_joint",
    "j3_joint",
    "j4_joint",
    "j5_joint",
    "j6_joint",
]


ACTUATOR_NAMES = [
    "j1_ctrl",
    "j2_ctrl",
    "j3_ctrl",
    "j4_ctrl",
    "j5_ctrl",
    "j6_ctrl",
]


EE_SITE_NAME = "ee_center_site"


# ============================================================
# 视频节奏
# ============================================================

INITIAL_PAUSE = 1.5
READY_PAUSE = 1.0

# Stage 2 演示时间
STAGE2_DURATION = 18.0

# 控制周期：200 Hz
CONTROL_DT = 0.005


# ============================================================
# Stage 1
#
# 不使用截图中的关节角。
# 直接使用世界坐标笛卡尔目标。
# ============================================================

READY_OFFSET = np.array([
    0.10,   # 世界 X：前移 10 cm
    0.00,
    0.18,   # 世界 Z：抬高 18 cm
], dtype=np.float64)


RAISE_KP = 7.0

RAISE_TIMEOUT = 8.0

RAISE_TOLERANCE = 0.003


# ============================================================
# Stage 2 Primary Task
#
# EE position PD
# ============================================================

# 不再使用之前非常暴力的 40~45
# 中等 Kp + 速度阻尼通常稳定得多
HOLD_KP = 16.0

HOLD_KD = 4.0


# Primary task 的 DLS 阻尼
PRIMARY_DAMPING = 0.025


# ============================================================
# Stage 2 Secondary Task
#
# 大幅度 Null-space self-motion
# ============================================================

# 追踪内部构型目标的速度
INTERNAL_KP = 2.2


# 大幅度目标。
#
# 注意：
# 这不是实际保证达到的关节振幅。
# 最后必须经过 Null Space 投影。
#
# rad:
# 1.0 rad ≈ 57°
# 1.5 rad ≈ 86°
# 2.0 rad ≈ 115°
#
INTERNAL_AMPLITUDE = np.array([
    1.65,
    1.30,
    1.55,
    2.00,
    1.65,
    2.20,
], dtype=np.float64)


# 不要太高频。
# 我们需要的是“大幅、看得清楚、很平滑”。
INTERNAL_FREQUENCY = np.array([
    0.48,
    0.40,
    0.57,
    0.66,
    0.51,
    0.73,
], dtype=np.float64)


INTERNAL_PHASE = np.array([
    0.00,
    1.10,
    2.00,
    0.50,
    2.60,
    1.40,
], dtype=np.float64)


# ============================================================
# 速度 / 安全限制
# ============================================================

# 足够快，但不要再像之前那样 4 rad/s 猛甩
MAX_JOINT_SPEED = 2.5


# 关节范围留一点余量
LIMIT_MARGIN = 0.10


# EE 偏差开始明显增大时，
# 自动削弱 secondary task
SOFT_EE_ERROR = 0.006     # 6 mm
HARD_EE_ERROR = 0.015     # 15 mm


# qdot 一阶低通滤波
#
# 越接近 1 越平滑
QDOT_FILTER_ALPHA = 0.82


# SVD 判断奇异值有效性的阈值
SVD_EPS = 1e-5


# ============================================================
# Helper
# ============================================================

def get_required_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(
        model,
        obj_type,
        name,
    )

    if obj_id < 0:
        raise RuntimeError(
            f"MuJoCo 模型中找不到：{name}"
        )

    return obj_id


def damped_pseudoinverse(J, damping):
    """
    Primary task 使用 Damped Least Squares：

        J# = J^T (J J^T + λ² I)^-1
    """

    task_dim = J.shape[0]

    A = (
        J @ J.T
        + damping ** 2
        * np.eye(task_dim)
    )

    return (
        J.T
        @ np.linalg.solve(
            A,
            np.eye(task_dim),
        )
    )


def exact_nullspace_projector(J):
    """
    使用 SVD 直接构造真正的 Null Space。

        J = U S V^T

    零空间由 V 中对应零奇异值的列张成。

    N = V_null V_null^T

    理论上：

        J N ≈ 0

    这比：

        I - J_damped# J

    更适合 secondary motion。
    """

    U, S, Vt = np.linalg.svd(
        J,
        full_matrices=True,
    )

    rank = int(
        np.sum(S > SVD_EPS)
    )

    V = Vt.T

    V_null = V[:, rank:]

    if V_null.shape[1] == 0:
        return np.zeros(
            (J.shape[1], J.shape[1])
        ), S, rank

    N = V_null @ V_null.T

    return N, S, rank


def clamp_qdot(qdot):
    return np.clip(
        qdot,
        -MAX_JOINT_SPEED,
        MAX_JOINT_SPEED,
    )


# ============================================================
# Load Model
# ============================================================

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(
    model
)


# ============================================================
# Resolve IDs
# ============================================================

joint_ids = np.array([
    get_required_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    for name in JOINT_NAMES
])


actuator_ids = np.array([
    get_required_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )
    for name in ACTUATOR_NAMES
])


ee_site_id = get_required_id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    EE_SITE_NAME,
)


qpos_indices = np.array([
    model.jnt_qposadr[joint_id]
    for joint_id in joint_ids
])


dof_indices = np.array([
    model.jnt_dofadr[joint_id]
    for joint_id in joint_ids
])


# ============================================================
# Safe joint range
# ============================================================

joint_lower = np.empty(6)
joint_upper = np.empty(6)


for i, (joint_id, actuator_id) in enumerate(
    zip(joint_ids, actuator_ids)
):

    j_lo, j_hi = (
        model.jnt_range[joint_id]
    )

    a_lo, a_hi = (
        model.actuator_ctrlrange[
            actuator_id
        ]
    )

    joint_lower[i] = (
        max(j_lo, a_lo)
        + LIMIT_MARGIN
    )

    joint_upper[i] = (
        min(j_hi, a_hi)
        - LIMIT_MARGIN
    )


joint_center = (
    0.5
    * (joint_lower + joint_upper)
)


joint_half_range = (
    0.5
    * (joint_upper - joint_lower)
)


# ============================================================
# Robot State
# ============================================================

def get_q():

    return (
        data.qpos[
            qpos_indices
        ].copy()
    )


def get_qvel():

    return (
        data.qvel[
            dof_indices
        ].copy()
    )


def get_ee_position():

    return (
        data.site_xpos[
            ee_site_id
        ].copy()
    )


def get_position_jacobian():

    jac_pos = np.zeros(
        (3, model.nv)
    )

    jac_rot = np.zeros(
        (3, model.nv)
    )

    mujoco.mj_jacSite(
        model,
        data,
        jac_pos,
        jac_rot,
        ee_site_id,
    )

    return (
        jac_pos[
            :,
            dof_indices
        ]
    )


def command_q(q_target):

    q_target = np.clip(
        q_target,
        joint_lower,
        joint_upper,
    )

    for actuator_id, value in zip(
        actuator_ids,
        q_target,
    ):

        data.ctrl[
            actuator_id
        ] = value


# ============================================================
# Physics / realtime
# ============================================================

def simulate_control_period(viewer):

    target_sim_time = (
        data.time
        + CONTROL_DT
    )

    while (
        data.time < target_sim_time
        and viewer.is_running()
    ):

        mujoco.mj_step(
            model,
            data,
        )


def realtime_wait(wall_start):

    elapsed = (
        time.perf_counter()
        - wall_start
    )

    remaining = (
        CONTROL_DT
        - elapsed
    )

    if remaining > 0:
        time.sleep(
            remaining
        )


# ============================================================
# Viewer reference marker
# ============================================================

def draw_lock_marker(
    viewer,
    position,
):
    """
    在被锁定 EE 坐标画一个较大的
    半透明青色球。

    红色 EE 应该始终保持在它的中心。
    """

    viewer.user_scn.ngeom = 1

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],

        mujoco.mjtGeom.mjGEOM_SPHERE,

        size=np.array([
            0.030,
            0.0,
            0.0,
        ]),

        pos=position,

        mat=np.eye(3).flatten(),

        rgba=np.array([
            0.0,
            1.0,
            1.0,
            0.20,
        ]),
    )


# ============================================================
# Init
# ============================================================

mujoco.mj_resetData(
    model,
    data,
)

mujoco.mj_forward(
    model,
    data,
)


initial_q = get_q()

initial_ee = (
    get_ee_position()
)


command_q(
    initial_q
)


ready_target = (
    initial_ee
    + READY_OFFSET
)


print()
print(
    "================================================"
)
print(
    "    Null-Space Motion — PPT Video Edition"
)
print(
    "================================================"
)

print()

print(
    "Initial EE:",
    np.round(initial_ee, 4)
)

print(
    "Ready target:",
    np.round(ready_target, 4)
)

print()


# ============================================================
# Viewer
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    viewer.cam.distance = 1.10

    viewer.cam.azimuth = 135

    viewer.cam.elevation = -18

    viewer.cam.lookat[:] = (
        initial_ee
    )


    # ========================================================
    # Stage 0
    #
    # 给录屏留一个干净的初始镜头
    # ========================================================

    print(
        "Stage 0: initial pose"
    )


    pause_start = (
        time.perf_counter()
    )

    while (
        viewer.is_running()
        and
        time.perf_counter()
        - pause_start
        < INITIAL_PAUSE
    ):

        wall_start = (
            time.perf_counter()
        )

        command_q(
            initial_q
        )

        simulate_control_period(
            viewer
        )

        viewer.sync()

        realtime_wait(
            wall_start
        )


    # ========================================================
    # Stage 1
    #
    # Cartesian raise
    # ========================================================

    print(
        "Stage 1: raise arm"
    )


    stage1_wall_start = (
        time.perf_counter()
    )

    stable_counter = 0


    qdot_filtered = np.zeros(6)


    while viewer.is_running():

        wall_start = (
            time.perf_counter()
        )


        mujoco.mj_forward(
            model,
            data,
        )


        q = get_q()

        ee_pos = (
            get_ee_position()
        )


        error = (
            ready_target
            - ee_pos
        )


        error_norm = (
            np.linalg.norm(
                error
            )
        )


        J = (
            get_position_jacobian()
        )


        J_primary = (
            damped_pseudoinverse(
                J,
                PRIMARY_DAMPING,
            )
        )


        qdot_primary = (
            J_primary
            @ (
                RAISE_KP
                * error
            )
        )


        # --------------------------------------------
        # Stage 1 secondary:
        #
        # 很轻的 joint-center preference
        # --------------------------------------------

        N, _, _ = (
            exact_nullspace_projector(
                J
            )
        )


        normalized_offset = (
            (q - joint_center)
            /
            np.maximum(
                joint_half_range,
                1e-6,
            )
        )


        qdot_center = (
            N
            @ (
                -0.25
                * normalized_offset
            )
        )


        qdot_raw = (
            qdot_primary
            + qdot_center
        )


        qdot_raw = (
            clamp_qdot(
                qdot_raw
            )
        )


        qdot_filtered = (
            QDOT_FILTER_ALPHA
            * qdot_filtered
            +
            (1.0 - QDOT_FILTER_ALPHA)
            * qdot_raw
        )


        q_command = (
            q
            +
            qdot_filtered
            * CONTROL_DT
        )


        command_q(
            q_command
        )


        simulate_control_period(
            viewer
        )


        viewer.sync()


        if (
            error_norm
            < RAISE_TOLERANCE
        ):

            stable_counter += 1

        else:

            stable_counter = 0


        if stable_counter >= 40:

            print(
                f"Stage 1 complete | "
                f"EE error = "
                f"{error_norm * 1000:.2f} mm"
            )

            break


        if (
            time.perf_counter()
            - stage1_wall_start
            > RAISE_TIMEOUT
        ):

            print(
                "Stage 1 timeout; "
                "continue from current pose."
            )

            break


        realtime_wait(
            wall_start
        )


    # ========================================================
    # Pause
    # ========================================================

    q_ready = get_q()


    pause_start = (
        time.perf_counter()
    )


    while (
        viewer.is_running()
        and
        time.perf_counter()
        - pause_start
        < READY_PAUSE
    ):

        wall_start = (
            time.perf_counter()
        )

        command_q(
            q_ready
        )

        simulate_control_period(
            viewer
        )

        viewer.sync()

        realtime_wait(
            wall_start
        )


    # ========================================================
    # Stage 2 Setup
    # ========================================================

    mujoco.mj_forward(
        model,
        data,
    )


    locked_ee = (
        get_ee_position()
    )


    null_start_q = (
        get_q()
    )


    viewer.cam.lookat[:] = (
        locked_ee
    )


    print()
    print(
        "Stage 2: LARGE smooth null-space motion"
    )

    print(
        "Locked EE:",
        np.round(
            locked_ee,
            4,
        )
    )

    print()


    stage2_sim_start = (
        data.time
    )

    stage2_wall_start = (
        time.perf_counter()
    )


    qdot_filtered = np.zeros(6)

    last_print = (
        data.time
    )


    # ========================================================
    # Stage 2
    # ========================================================

    while viewer.is_running():

        wall_start = (
            time.perf_counter()
        )


        mujoco.mj_forward(
            model,
            data,
        )


        q = get_q()

        qvel = get_qvel()

        ee_pos = (
            get_ee_position()
        )


        J = (
            get_position_jacobian()
        )


        # ====================================================
        # EE actual Cartesian velocity
        # ====================================================

        ee_velocity = (
            J @ qvel
        )


        position_error = (
            locked_ee
            - ee_pos
        )


        error_norm = (
            np.linalg.norm(
                position_error
            )
        )


        # ====================================================
        # PRIMARY
        #
        # Cartesian position PD
        #
        # xdot_cmd =
        #
        #     Kp (xd - x)
        #     - Kd xdot
        #
        # ====================================================

        desired_ee_velocity = (
            HOLD_KP
            * position_error
            -
            HOLD_KD
            * ee_velocity
        )


        J_primary = (
            damped_pseudoinverse(
                J,
                PRIMARY_DAMPING,
            )
        )


        qdot_primary = (
            J_primary
            @ desired_ee_velocity
        )


        # ====================================================
        # EXACT NULL SPACE
        # ====================================================

        N, singular_values, rank = (
            exact_nullspace_projector(
                J
            )
        )


        # ====================================================
        # SECONDARY TARGET
        #
        # 大幅度、慢速、可视化效果清楚
        # ====================================================

        t = (
            data.time
            - stage2_sim_start
        )


        oscillation = (
            INTERNAL_AMPLITUDE
            *
            np.sin(
                INTERNAL_FREQUENCY
                * t
                +
                INTERNAL_PHASE
            )
        )


        desired_internal_q = (
            null_start_q
            + oscillation
        )


        desired_internal_q = (
            np.clip(
                desired_internal_q,
                joint_lower,
                joint_upper,
            )
        )


        internal_error = (
            desired_internal_q
            - q
        )


        desired_internal_velocity = (
            INTERNAL_KP
            * internal_error
        )


        # ====================================================
        # Joint limit avoidance
        # ====================================================

        normalized_offset = (
            (q - joint_center)
            /
            np.maximum(
                joint_half_range,
                1e-6,
            )
        )


        desired_internal_velocity += (
            -0.18
            * normalized_offset
        )


        # ====================================================
        # Project into exact null space
        # ====================================================

        qdot_secondary = (
            N
            @ desired_internal_velocity
        )


        # ====================================================
        # EE error gating
        #
        # < 6 mm:
        #   100% secondary
        #
        # 6 ~ 15 mm:
        #   逐渐减弱
        #
        # > 15 mm:
        #   secondary = 0
        # ====================================================

        if (
            error_norm
            <= SOFT_EE_ERROR
        ):

            secondary_scale = 1.0

        elif (
            error_norm
            >= HARD_EE_ERROR
        ):

            secondary_scale = 0.0

        else:

            secondary_scale = (
                1.0
                -
                (
                    error_norm
                    - SOFT_EE_ERROR
                )
                /
                (
                    HARD_EE_ERROR
                    - SOFT_EE_ERROR
                )
            )


        qdot_raw = (
            qdot_primary
            +
            secondary_scale
            * qdot_secondary
        )


        qdot_raw = (
            clamp_qdot(
                qdot_raw
            )
        )


        # ====================================================
        # Low-pass filter
        #
        # 去掉高频关节目标跳变
        # ====================================================

        qdot_filtered = (
            QDOT_FILTER_ALPHA
            * qdot_filtered
            +
            (1.0 - QDOT_FILTER_ALPHA)
            * qdot_raw
        )


        # ====================================================
        # Position actuator command
        # ====================================================

        q_command = (
            q
            +
            qdot_filtered
            * CONTROL_DT
        )


        command_q(
            q_command
        )


        # ====================================================
        # Physics
        # ====================================================

        simulate_control_period(
            viewer
        )


        # 青色世界坐标锁定标记
        draw_lock_marker(
            viewer,
            locked_ee,
        )


        viewer.sync()


        # ====================================================
        # Console
        # ====================================================

        if (
            data.time
            - last_print
            >= 0.5
        ):

            last_print = (
                data.time
            )


            current_ee = (
                get_ee_position()
            )


            drift_mm = (
                np.linalg.norm(
                    current_ee
                    - locked_ee
                )
                * 1000.0
            )


            print(
                f"EE drift "
                f"{drift_mm:6.2f} mm"
                f" | null dim "
                f"{6-rank}"
                f" | secondary "
                f"{secondary_scale:4.2f}"
                f" | q "
                f"{np.round(q, 2)}"
            )


        # ====================================================
        # 视频结束
        # ====================================================

        if (
            time.perf_counter()
            - stage2_wall_start
            >= STAGE2_DURATION
        ):

            print()
            print(
                "Stage 2 finished."
            )

            break


        realtime_wait(
            wall_start
        )


    # ========================================================
    # Ending pause
    # ========================================================

    ending_q = get_q()

    ending_start = (
        time.perf_counter()
    )


    while (
        viewer.is_running()
        and
        time.perf_counter()
        - ending_start
        < 1.5
    ):

        wall_start = (
            time.perf_counter()
        )

        command_q(
            ending_q
        )

        draw_lock_marker(
            viewer,
            locked_ee,
        )

        simulate_control_period(
            viewer
        )

        viewer.sync()

        realtime_wait(
            wall_start
        )