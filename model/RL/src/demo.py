import time
import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('./model/mjcf/arm_mjcf.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 300:  # 改为300秒（5分钟）
        step_start = time.time()
        
        # 可选：施加随机控制信号
        # d.ctrl[:] = np.random.uniform(-1, 1, m.nu)
        
        mujoco.mj_step(m, d)

        
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        
        viewer.sync()
        
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
           