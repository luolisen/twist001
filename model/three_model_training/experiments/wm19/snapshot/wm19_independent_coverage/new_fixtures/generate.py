"""Preregistered physical challenge fixtures. No model or task acceptance."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wm_consistent_v7'))
from physical_labels import np,mujoco,setup,render,public_motion,sha,atomic,snap,restore,segmentation_labels,geometry,physical_outcome,CONTACT_TYPES
ROOT=Path(__file__).resolve().parent
from blind_scenes import ik
from layout_sampler import sample
from scipy.spatial.transform import Rotation
import json, os

def main():
    plan=json.loads((ROOT/'execution_plan.json').read_text())
    out=ROOT/'coverage_fixtures';out.mkdir(exist_ok=False)
    source=json.loads((ROOT.parents[1]/'wm_contact_v5/input_plan.json').read_text())['sources'][0]
    template=ROOT.parents[1]/'wm_contact_v5/sources'/source['source_group']/'scene.xml'
    assert sha(template)==source['portable_scene_sha256']
    kinds=('object_nonfinger','object_object','object_tray_wall','robot_tray','robot_floor')
    records=[];fail=[]
    for seed in plan['coverage_seeds']:
        for kind in kinds:
            for proximity in ('near','far'):
                name=f'{seed}_{kind}_{proximity}';m,d,ix=setup(template)
                renderer=mujoco.Renderer(m,360,640);rng=np.random.default_rng(seed)
                try:
                    scatter,_=sample(seed,[.08,.33],8,selected_indices=[0,2,4]);assert scatter is not None
                    d.qpos[ix.q]=[0,-.05,.1,0,0,0];d.qpos[ix.lq]=-.02;d.qpos[ix.rq]=.02
                    d.ctrl[ix.aids]=[0,-.05,.1,0,0,0,.04]
                    for i in range(8):
                        j=m.joint('pick_cube_free' if i==0 else f'cube_free_{i}').id
                        a=int(m.jnt_qposadr[j]);d.qpos[a:a+7]=[*scatter[i,:2],.015,1,0,0,0]
                    m.light_diffuse[0]=np.array([.5]*3)*rng.uniform(.85,1.15);mujoco.mj_forward(m,d)
                    j=m.joint('pick_cube_free').id;a=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j])
                    if kind=='object_object':
                        j2=m.joint('cube_free_1').id;a2=int(m.jnt_qposadr[j2])
                        center=np.array([.24,.20])+rng.uniform(-.015,.015,2)
                        distance=.051 if proximity=='near' else .095
                        d.qpos[a:a+3]=[*center,.015];d.qpos[a2:a2+3]=[center[0]+distance,center[1],.015]
                        d.qvel[v]=.22 if proximity=='near' else 0
                    elif kind=='object_tray_wall':
                        wall=m.geom('tray_wall_x0').id;center=d.geom_xpos[wall].copy()
                        distance=.0375 if proximity=='near' else .065
                        d.qpos[a:a+3]=[center[0]-distance,center[1]+rng.uniform(-.018,.018),.015]
                        d.qvel[v]=.22 if proximity=='near' else 0
                    elif kind=='object_nonfinger':
                        base=m.body('base_link').id
                        candidates=[g for g in range(m.ngeom) if int(m.geom_bodyid[g])==base and int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_MESH) and (m.geom_contype[g] or m.geom_conaffinity[g])]
                        assert len(candidates)==1
                        g=candidates[0];mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
                        verts=m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                        lo,hi=verts.min(0),verts.max(0);center=(lo+hi)/2
                        distance=.036 if proximity=='near' else .065
                        d.qpos[a:a+3]=[hi[0]+distance,center[1]+rng.uniform(-.02,.02),.015]
                        d.qvel[v]=-.22 if proximity=='near' else 0
                    else:
                        bottom=d.geom_xpos[m.geom('tray_bottom').id].copy()
                        goal=bottom+np.array([rng.uniform(-.04,.04),rng.uniform(-.012,.005) if kind=='robot_tray' and proximity=='near' else rng.uniform(-.02,.02),.055 if kind=='robot_tray' and proximity=='near' else .075 if proximity=='near' else .12])
                        if kind=='robot_tray' and proximity=='far':goal=np.array([.08+rng.uniform(-.025,.025),.26+rng.uniform(-.012,.012),.075])
                        if kind=='robot_floor':goal=np.array([.22+rng.uniform(-.03,.03),.21+rng.uniform(-.02,.02),.075 if proximity=='near' else .12])
                        rotation=Rotation.from_euler('z',-np.pi/2).as_matrix()@Rotation.from_euler('x',np.pi).as_matrix()
                        q=ik(m,ix,goal,rotation,d.qpos[ix.q].copy())
                        d.qpos[ix.q]=q;d.ctrl[ix.aids]=np.r_[q,.04]
                    # Causal flight permits observable motion before later contact; no applied future impulses.
                    if kind in ('object_nonfinger','object_object','object_tray_wall'):d.qpos[a+2]=.065
                    mujoco.mj_forward(m,d);images=render(m,d,renderer)[1];state=public_motion(m,d,ix)
                    actions=np.repeat(d.ctrl[ix.aids][None,:],8,axis=0).astype('float32')
                    if kind in ('robot_tray','robot_floor') and proximity=='near':
                        low=goal.copy();low[2]-=.05 if kind=='robot_tray' else .07
                        target=ik(m,ix,low,rotation,d.qpos[ix.q].copy())
                        for step in range(8):actions[step,:6]=q+(step+1)/8*(target-q)
                    ids=segmentation_labels(m,d,renderer);baseline=snap(m,d)
                    truth=physical_outcome(m,d,ix,geometry(m),actions,renderer,ids)
                    path=out/(name+'.npz');np.savez_compressed(path,images=images,state=state,actions=actions,integration_state=baseline,**truth)
                    records.append(dict(seed=seed,fixture_family=kind,proximity=proximity,cache=str(path.relative_to(ROOT)),sha256=sha(path),scene=str(template),private_geometry_is_generation_or_truth_only=True))
                except Exception as e:fail.append(dict(seed=seed,fixture_family=kind,proximity=proximity,error=repr(e)))
                finally:
                    renderer.close();atomic(ROOT/'coverage_status.json',dict(stage='wm13_new_physical_fixture_generation',cases_done=len(records)+len(fail),cases_total=len(plan['coverage_seeds'])*10,records=len(records),failures=len(fail),pid=os.getpid(),compute_host='M2 Max'))
        print('coverage seed',seed,'records',len(records),'failures',len(fail),flush=True)
    counts={k:{'positives':0,'negatives':0,'new_positives':0} for k in kinds}
    for row in records:
        with np.load(ROOT/row['cache']) as a:
            for k in kinds:
                index=CONTACT_TYPES.index(k);y=bool(a['target_contacts'][index]);counts[k]['positives']+=int(y);counts[k]['negatives']+=int(not y);counts[k]['new_positives']+=int(a['target_new_contacts'][index])
    missing=[k for k in kinds if counts[k]['positives']<3 or counts[k]['negatives']<3]
    report=dict(records=records,failures=fail,coverage_counts=counts,coverage_gate_passed=not missing and not fail,missing_classes=missing,no_resampling=True,fitting_allowed=False,execution_plan_sha256=sha(ROOT/'execution_plan.json'),scope='Targeted synthetic physical fixtures, not policy rollouts or full task success. Robot self-contact and retention are not covered by this gate.',hardware_enabled=False,full_task_acceptance_passed=False)
    atomic(ROOT/'coverage_manifest.json',report)
    # Preserve insufficient coverage; final combined gate blocks fitting without resampling.

if __name__=='__main__':main()
