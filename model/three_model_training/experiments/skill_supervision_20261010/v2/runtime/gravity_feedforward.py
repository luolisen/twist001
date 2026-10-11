"""Public-encoder fixed-structure gravity feedforward; no load/contact input."""
import hashlib,json
from pathlib import Path
import numpy as np
import mujoco

def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

class FixedStructureGravity:
    def __init__(self,scene):
        self.method_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.m=mujoco.MjModel.from_xml_path(str(scene));self.d=mujoco.MjData(self.m)
        m=self.m
        self.aids=np.array([m.actuator(f'j{i}_ctrl').id for i in range(1,7)])
        self.jids=m.actuator_trnid[self.aids,0];self.qa=m.jnt_qposadr[self.jids];self.va=m.jnt_dofadr[self.jids]
        if not (np.all(m.actuator_gainprm[self.aids,0]==100) and
                np.all(m.actuator_biasprm[self.aids,:3]==np.array([0,-100,-10])) and
                np.all(m.actuator_gear[self.aids]==np.array([1,0,0,0,0,0])) and
                np.all(m.actuator_dyntype[self.aids]==0)):
            raise RuntimeError('unsupported_actuator_law')
        root=m.body('measured_robot_mount').id;self.fixed=[];self.excluded=[]
        for b in range(1,m.nbody):
            ancestors=[];c=b
            while c:ancestors.append(c);c=int(m.body_parentid[c])
            if root not in ancestors:continue
            movable=any(int(m.jnt_type[j])!=int(mujoco.mjtJoint.mjJNT_HINGE)
                for x in ancestors for j in range(m.body_jntadr[x],m.body_jntadr[x]+m.body_jntnum[x]))
            (self.excluded if movable else self.fixed).append(b)
        self.parameters={'body_names':[m.body(b).name for b in self.fixed],
            'excluded_body_names':[m.body(b).name for b in self.excluded],
            'gravity_m_s2':m.opt.gravity.tolist(),
            'mass_kg':m.body_mass[self.fixed].tolist(),'body_ipos_m':m.body_ipos[self.fixed].tolist(),
            'body_pos_m':m.body_pos[self.fixed].tolist(),'body_quat':m.body_quat[self.fixed].tolist(),
            'parent_ids':m.body_parentid[self.fixed].tolist(),'joint_names':[m.joint(int(j)).name for j in self.jids],
            'joint_axis':m.jnt_axis[self.jids].tolist(),'joint_pos':m.jnt_pos[self.jids].tolist(),
            'qpos0':m.qpos0[self.qa].tolist(),'kp':100,'kv':10,
            'actuator_force_limits_Nm':m.actuator_forcerange[self.aids].tolist()}
        self.parameters_sha256=hashlib.sha256(json.dumps(self.parameters,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.calls=0

    def known_gravity(self,q_public):
        q=np.asarray(q_public)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('invalid_public_encoder')
        limits=self.m.jnt_range[self.jids]
        if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError('public_encoder_mechanical_range')
        # All non-arm states remain model defaults. Excluded bodies contribute
        # no COM Jacobian term; neither objects nor slide truth is read.
        self.d.qpos[self.qa]=q;self.d.qvel[:]=0
        mujoco.mj_fwdPosition(self.m,self.d)
        g=np.zeros(self.m.nv)
        for b in self.fixed:
            jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacBodyCom(self.m,self.d,jp,jr,b)
            g-=jp.T@(self.m.body_mass[b]*self.m.opt.gravity)
        self.calls+=1
        return g[self.va].copy()

    def generate(self,public_state,nominal,index,t,capture_id):
        s=np.asarray(public_state);n=np.asarray(nominal)
        if s.shape!=(21,) or not np.isfinite(s).all() or n.shape!=(7,) or n.dtype!=np.float32 or not np.isfinite(n).all():
            raise ValueError('invalid_feedforward_input')
        g=self.known_gravity(s[:6]);u=n.astype(float);u[:6]+=g/100;u=u.astype(np.float32)
        if not np.isfinite(u).all():raise ValueError('nonfinite_feedforward_output')
        return {'command_index':int(index),'public_time_s':float(t),'q_public_rad':s[:6].tolist(),
            'public_state':s.tolist(),'public_state_sha256':digest(s),'public_capture_id':capture_id,
            'nominal_command':n.tolist(),'nominal_sha256':digest(n),
            'known_gravity_nm':g.tolist(),'servo_command':u.tolist(),'servo_sha256':digest(u),
            'method_sha256':self.method_sha256,'parameters_sha256':self.parameters_sha256,
            'law':'u_arm=q_nom+g_known(q_public)/100; clamp unchanged',
            'unknown_payload_compensated':False,'private_contact_or_object_input':False}
