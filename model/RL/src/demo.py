"""REV6 simulation viewer and bounded offline camera check."""
from pathlib import Path
import argparse
import json
import time
import mujoco
import numpy as np
from PIL import Image
from preprocess_wrist_camera import preprocess_simulated_wrist_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('arm', 'task'), default='arm')
    parser.add_argument('--check', type=Path, help='Save offline camera images and load-check report')
    args = parser.parse_args()
    name = 'arm_mjcf.xml' if args.scene == 'arm' else 'task_scene.xml'
    path = Path(__file__).resolve().parent / 'model/mjcf' / name
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    initial = [0, -.05, .10, 0, 0, 0]
    for i, value in enumerate(initial, 1):
        data.qpos[model.jnt_qposadr[model.joint(f'j{i}_joint').id]] = value
        data.ctrl[model.actuator(f'j{i}_ctrl').id] = value
    mujoco.mj_forward(model, data)
    if args.check:
        args.check.mkdir(parents=True, exist_ok=True)
        report = {'mujoco_version': mujoco.__version__, 'scene': args.scene,
                  'joints': model.njnt, 'actuators': model.nu, 'cameras': {},
                  'scope': 'Load, render and 100-step numerical smoke check only; no task success or physical acceptance.'}
        with mujoco.Renderer(model, height=360, width=640) as renderer:
            for camera, size in [('front_rgb', 256), ('wrist_camera', 512)]:
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render().copy()
                policy = preprocess_simulated_wrist_frame(rgb, policy_size=size)
                assert rgb.shape == (360, 640, 3) and rgb.std() > 0
                pad = (size - size * 9 // 16) // 2
                assert not policy[:pad].any() and not policy[-pad:].any()
                Image.fromarray(rgb).save(args.check / f'{camera}_native.png')
                Image.fromarray(policy).save(args.check / f'{camera}_policy.png')
                report['cameras'][camera] = {'fovy': float(model.cam_fovy[model.camera(camera).id]),
                                            'render_shape': list(rgb.shape), 'policy_shape': list(policy.shape)}
        for _ in range(100):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert not any(w.number for w in data.warning)
        report['finite_100_steps'] = True
        (args.check / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
