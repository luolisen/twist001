"""Task-independent simulation labels. Geometry and segmentation are targets only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / 'wm_contact_v5'))
from physics import setup, public_motion, render, snap, restore, sha, atomic
import numpy as np
import mujoco
from PIL import Image

CONTACT_TYPES = ('object_finger', 'object_nonfinger', 'object_object',
                 'object_tray_wall', 'object_tray_bottom', 'object_floor',
                 'robot_tray', 'robot_floor', 'robot_self')

def geometry(m):
    objects = {m.geom('pick_cube_geom' if i == 0 else f'cube_geom_{i}').id for i in range(8)}
    bodies = {int(m.geom_bodyid[g]) for g in objects}
    fingers = {m.geom('left_finger_collision').id, m.geom('right_finger_collision').id}
    floor = m.geom('floor').id
    bottom = m.geom('tray_bottom').id
    walls = {m.geom(f'tray_wall_x{i}').id for i in range(4)} | {m.geom(f'tray_wall_y{i}').id for i in range(2)}
    robot = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in bodies | {0, m.body('tray').id}
             and (m.geom_contype[g] or m.geom_conaffinity[g])}
    return dict(objects=objects, fingers=fingers, floor=floor, bottom=bottom, walls=walls, robot=robot)

def category(g, a, b):
    pair = {a, b}
    obj = pair & g['objects']
    if len(obj) == 2:
        return 2
    if obj:
        other = next(iter(pair - obj)) if len(pair - obj) else a
        if other in g['fingers']: return 0
        if other == g['bottom']: return 4
        if other in g['walls']: return 3
        if other == g['floor']: return 5
        if other in g['robot']: return 1
        return None
    if pair & g['robot']:
        if pair & (g['walls'] | {g['bottom']}): return 6
        if g['floor'] in pair: return 7
        if pair <= g['robot']: return 8
    return None

def segmentation_labels(m, d, renderer):
    """Initial visible geom footprints for supervision; never passed to model."""
    result = []
    for camera, side in [('front_rgb', 256), ('wrist_camera', 512)]:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d, camera=camera)
        s = renderer.render().copy()
        assert s.shape == (360, 640, 2)
        ids = np.where(s[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[:, :, 0], -1).astype('int32')
        h = side * 9 // 16
        pad = (side - h) // 2
        square = np.full((side, side), -1, np.int32)
        square[pad:pad+h] = np.asarray(Image.fromarray(ids).resize((side, h), Image.Resampling.NEAREST))
        result.append(np.asarray(Image.fromarray(square).resize((128, 128), Image.Resampling.NEAREST)))
    renderer.disable_segmentation_rendering()
    return np.stack(result)

def footprint(ids, geoms):
    mask = np.isin(ids, list(geoms))
    return mask.reshape(2, 32, 4, 32, 4).max(axis=(2, 4)).astype('uint8')

def body_footprint(m, ids, geoms):
    """Map collision shapes to visible shapes on the same rigid body, targets only."""
    bodies = {int(m.geom_bodyid[g]) for g in geoms if int(m.geom_bodyid[g]) != 0}
    visible_geoms = set(geoms) | {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies}
    return footprint(ids, visible_geoms)

def physical_outcome(m, d, ix, g, actions, renderer, ids):
    """No logical target, stage, assignment, task or object ID is an inference input."""
    start = public_motion(m, d, ix)[:7].copy()
    ee0 = d.site_xpos[ix.ee].copy()
    initial = {tuple(sorted((int(c.geom1), int(c.geom2)))) for c in d.contact if c.dist <= 0}
    touched = [set() for _ in CONTACT_TYPES]
    events = np.zeros(9, np.uint8)
    new_events = events.copy()
    start_grasp = set()
    def grasped():
        pairs = [{int(c.geom1), int(c.geom2)} for c in d.contact if c.dist <= 0]
        return {obj for obj in g['objects'] if all({obj, finger} in pairs for finger in g['fingers'])
                and not any(obj in pair and pair & {g['floor'], g['bottom']} for pair in pairs)}
    start_grasp = grasped()
    for action in actions:
        d.ctrl[ix.aids] = action
        for _ in range(40):
            mujoco.mj_step(m, d)
            for c in d.contact:
                if c.dist > 0: continue
                a, b = int(c.geom1), int(c.geom2)
                k = category(g, a, b)
                if k is None: continue
                events[k] = 1
                if tuple(sorted((a, b))) not in initial: new_events[k] = 1
                # Mark initial visible object/link region, not a fictitious contact point.
                touched[k].update(({a, b} & g['objects']) or ({a, b} & g['robot']))
    mujoco.mj_forward(m, d)
    terminal_grasp = grasped()
    maps = np.stack([body_footprint(m, ids, geoms) for geoms in touched], axis=1)
    hidden = np.array([bool(events[k] and not maps[:, k].any()) for k in range(9)], np.uint8)
    retention = np.array([bool(terminal_grasp), bool(start_grasp - terminal_grasp)], np.uint8)
    grasp_map = body_footprint(m, ids, terminal_grasp)
    _, future = render(m, d, renderer)
    delta = np.r_[public_motion(m, d, ix)[:7] - start, d.site_xpos[ix.ee] - ee0].astype('float32')
    return dict(target_motion_delta=delta, target_future_rgb=future,
                target_contacts=events, target_new_contacts=new_events,
                target_contact_regions=maps, target_unlocalized_contacts=hidden,
                target_retention=retention, target_grasp_region=grasp_map)
