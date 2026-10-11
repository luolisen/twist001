"""Registered static-geometry protection adapter, isolated from policy inputs.

Only actual public q is updated. Other scene geometry stays at its registered
layout. This is not an object tracker, carry detector, or future physics rollout.
Original live simulator collision/displacement protection remains independent.
"""
import itertools
import numpy as np

def corners(lo,hi):return np.array(list(itertools.product(*zip(lo,hi))),float)
def overlaps(a,b):return bool(np.all(a[0]<=b[1]) and np.all(b[0]<=a[1]))

class StaticTransportGeometry:
    def __init__(self,contract,scene,calibration,labels):
        from skill_generators import fk_pose
        self.fk=fk_pose;self.cal=calibration;self.c=contract;self.labels=labels
        self.m,self.d,self.ix=labels.setup(scene)
        for i in range(6):
            if not np.array_equal(self.m.joint('j'+str(i+1)+'_joint').range,np.asarray(contract['mechanical_ranges'][i])):
                raise RuntimeError('loaded_mechanical_range_differs_from_contract')
            if not np.array_equal(self.m.actuator('j'+str(i+1)+'_ctrl').ctrlrange,np.asarray(contract['control_ranges'][i])):
                raise RuntimeError('loaded_control_range_differs_from_contract')
        self.slide_configs=list(itertools.product(*[self.m.joint(n).range.tolist() for n in ('left_slide','right_slide')]))
        # Do not restore actual object poses. Use static registered scene layout.
        self.geo=labels.geometry(self.m);self.target=self.m.geom('pick_cube_geom').id
        self.d.qpos[self.ix.q]=np.asarray(contract['initial_q_rad'])
        labels.mujoco.mj_forward(self.m,self.d)
        self.rotation0=self.fk(self.cal,contract['initial_q_rad'])[:3,:3]
        tool=int(self.m.body_weldid[self.m.site_bodyid[self.ix.ee]])
        self.obstacles=(self.geo['objects']-{self.target})|self.geo['walls']|{self.geo['bottom'],self.geo['floor']}
        self.robot_obstacles={g for g in self.geo['robot']-self.geo['fingers'] if int(self.m.body_weldid[self.m.geom_bodyid[g]])!=tool}
        p=contract['conditional_payload'];self.env=corners(p['payload_min_m'],p['payload_max_m'])
        self.minimum_clearance=float('inf');self.calls=0

    def aabb(self,g):
        m,d=self.m,self.d;j=self.labels.mujoco;k=int(m.geom_type[g]);size=m.geom_size[g]
        if k==int(j.mjtGeom.mjGEOM_PLANE):return None
        if k==int(j.mjtGeom.mjGEOM_MESH):
            mesh=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh]);v=m.mesh_vert[a:a+n]
        else:
            types=j.mjtGeom
            if k==int(types.mjGEOM_SPHERE):extent=np.repeat(size[0],3)
            elif k==int(types.mjGEOM_CAPSULE):extent=np.array([size[0],size[0],size[0]+size[1]])
            elif k==int(types.mjGEOM_CYLINDER):extent=np.array([size[0],size[0],size[1]])
            elif k==int(types.mjGEOM_BOX):extent=size
            else:raise RuntimeError('unsupported_registered_geometry')
            v=corners(-extent,extent)
        pts=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        return np.stack([pts.min(0),pts.max(0)])

    def __call__(self,q):
        m,d,l=self.m,self.d,self.labels
        self.calls+=1;d.qpos[self.ix.q]=q;l.mujoco.mj_forward(m,d)
        pose=self.fk(self.cal,q);fail=[]
        from scipy.spatial.transform import Rotation
        if Rotation.from_matrix(self.rotation0@pose[:3,:3].T).magnitude()>self.c['rotation_tolerance_rad']:
            fail.append('task_orientation_constraint')
        # Public gap does not identify the two individual slider positions.
        # Audit both full-range endpoint combinations, without assuming symmetry.
        # This remains finite static sampling, not a swept-volume certificate.
        for left,right in self.slide_configs:
            d.qpos[m.joint('left_slide').qposadr[0]]=left
            d.qpos[m.joint('right_slide').qposadr[0]]=right
            l.mujoco.mj_forward(m,d)
            for c in d.contact:
                if c.dist>0:continue
                a,b=int(c.geom1),int(c.geom2)
                if self.target in (a,b):continue
                k=l.category(self.geo,a,b)
                if k==8:
                    ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
                    adjacent=int(m.body_parentid[ba])==bb or int(m.body_parentid[bb])==ba
                    if ba!=bb and not adjacent and m.body_weldid[ba]!=m.body_weldid[bb]:fail.append('nonadjacent_robot_collision')
                if k in (1,6,7):fail.append('robot_registered_environment_collision')
        payload=self.env@pose[:3,:3].T+pose[:3,3];box=np.stack([payload.min(0),payload.max(0)])
        for g in sorted(self.obstacles|self.robot_obstacles):
            obstacle=self.aabb(g)
            if obstacle is None:
                clearance=float(box[0,2]-d.geom_xpos[g,2]);hit=clearance<=0
            else:
                clearance=float(np.linalg.norm(np.maximum(np.maximum(obstacle[0]-box[1],box[0]-obstacle[1]),0)))
                hit=overlaps(box,obstacle)
            self.minimum_clearance=min(self.minimum_clearance,clearance)
            if hit:fail.append('conditional_payload_registered_geometry_overlap')
        return fail

def additional_live_transport_contacts(m,d,geo,labels,carried):
    """Independent simulator physical-protection channel, never Jev/public evidence.

    Adds actual robot-floor/tray and carried-object environmental contacts which
    were only logged in the old executor. Original stop rules remain upstream.
    """
    for contact in d.contact:
        if contact.dist>0:continue
        a,b=int(contact.geom1),int(contact.geom2);k=labels.category(geo,a,b)
        if k in (6,7):return 'transport_robot_environment_contact'
        if carried in (a,b):
            other=b if a==carried else a
            if other not in geo['fingers']:return 'transport_carried_object_environment_contact'
    return None
