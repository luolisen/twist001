"""Portable MuJoCo evaluator. Privileged geometry is restricted to labels/evaluation."""
import os
os.environ.setdefault('MUJOCO_GL','cgl')
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,numpy as np,mujoco
from PIL import Image
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def setup(path):
 m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);assert m.nq==64 and m.nu==7 and abs(m.opt.timestep-.001)<1e-12
 aids=np.array([m.actuator(f'j{i}_ctrl').id for i in range(1,7)]+[m.actuator('g').id]);joint=np.array([m.actuator(f'j{i}_ctrl').trnid[0] for i in range(1,7)]);q=np.array([m.jnt_qposadr[j] for j in joint]);v=np.array([m.jnt_dofadr[j] for j in joint]);lj=m.joint('left_slide').id;rj=m.joint('right_slide').id
 ix=SimpleNamespace(aids=aids,q=q,v=v,lq=int(m.jnt_qposadr[lj]),rq=int(m.jnt_qposadr[rj]),lv=int(m.jnt_dofadr[lj]),rv=int(m.jnt_dofadr[rj]),ee=m.site('ee_center_site').id)
 return m,d,ix
def public_motion(m,d,ix):
 return np.r_[d.qpos[ix.q],d.qpos[ix.rq]-d.qpos[ix.lq],d.qvel[ix.v],d.qvel[ix.rv]-d.qvel[ix.lv],d.ctrl[ix.aids]].astype('float32')
def context(m,active,stage,assigned=()):
 bodies=[m.body('pick_cube' if i==0 else f'cube_{i}').id for i in range(8)]
 return SimpleNamespace(model=m,geoms=[m.geom('pick_cube_geom' if i==0 else f'cube_geom_{i}').id for i in range(8)],floor_geom=m.geom('floor').id,bottom=m.geom('tray_bottom').id,walls={m.geom(f'tray_wall_x{i}').id for i in range(4)}|{m.geom(f'tray_wall_y{i}').id for i in range(2)},left_geom=m.geom('left_finger_collision').id,right_geom=m.geom('right_finger_collision').id,robot={i for i in range(m.ngeom) if int(m.geom_bodyid[i]) not in set(bodies)|{0,m.body('tray').id} and (m.geom_contype[i] or m.geom_conaffinity[i])},active=active,stage=stage,assigned=set(assigned))
def render(m,d,renderer):
 full=[];small=[]
 for camera,side in [('front_rgb',256),('wrist_camera',512)]:
  renderer.disable_segmentation_rendering();renderer.update_scene(d,camera=camera);rgb=renderer.render().copy();h=side*9//16;pad=(side-h)//2;im=np.zeros((side,side,3),np.uint8);im[pad:pad+h]=np.asarray(Image.fromarray(rgb).resize((side,h),Image.Resampling.LANCZOS));full.append(im);small.append(np.asarray(Image.fromarray(im).resize((128,128),Image.Resampling.LANCZOS)))
 return full,np.array(small)
def snap(m,d):
 flag=mujoco.mjtState.mjSTATE_INTEGRATION;a=np.zeros(mujoco.mj_stateSize(m,flag));mujoco.mj_getState(m,d,a,flag);return a
def restore(m,d,a):mujoco.mj_setState(m,d,a,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
def outcome(m,d,ix,ctx,actions):
 start=public_motion(m,d,ix)[:7];ee0=d.site_xpos[ix.ee].copy();body=m.geom_bodyid[ctx.geoms[ctx.active]];obj0=d.xpos[body].copy();kinds=set()
 for action in actions:
  d.ctrl[ix.aids]=action
  for _ in range(40):
   mujoco.mj_step(m,d)
   for c in d.contact:
    kind=classify(ctx,int(c.geom1),int(c.geom2))
    if kind:kinds.add(kind)
 mujoco.mj_forward(m,d);end=public_motion(m,d,ix)[:7];pairs=[{int(c.geom1),int(c.geom2)} for c in d.contact];gid=ctx.geoms[ctx.active];grasp={gid,ctx.left_geom} in pairs and {gid,ctx.right_geom} in pairs and not any(gid in p and bool(p&{ctx.floor_geom,ctx.bottom}) for p in pairs)
 delta=np.r_[end-start,d.site_xpos[ix.ee]-ee0,d.xpos[body]-obj0].astype('float32');events=np.array([grasp,bool(kinds)],np.float32);return delta,events,sorted(kinds)

def classify(self,a,b):
        pair={a,b};cube=[i for i,x in enumerate(self.geoms) if x in pair]
        if cube:
            if len(cube)>1:return 'cube_cube'
            i=cube[0];other=b if a==self.geoms[i] else a
            if other==self.floor_geom:return None
            if other==self.bottom and (i in self.assigned or (self.active==i and self.stage in ('place','release','withdraw','settle'))):return None
            if other in (self.left_geom,self.right_geom) and i==self.active:return None
            return 'cube_wall' if other in self.walls else 'protected_cube_robot' if i!=self.active else 'cube_nonfinger'
        if a in self.robot or b in self.robot:
            if pair<={self.floor_geom,self.bottom}|self.walls:return None
            if self.floor_geom in pair:
                rb=b if a==self.floor_geom else a;body=self.model.body(int(self.model.geom_bodyid[rb])).name
                return None if body in ('base_link','j1_Link') else 'robot_table'
            if self.bottom in pair or bool(pair&self.walls):return 'robot_tray'
            names={self.model.body(int(self.model.geom_bodyid[x])).name for x in pair}
            return None if names=={'base_link','j1_Link'} else 'robot_self'
        return None
