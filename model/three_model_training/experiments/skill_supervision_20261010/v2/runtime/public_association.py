"""Public RGB regional association with static camera calibration and encoder FK.
No simulator data, object geometry, segmentation or contact input.
"""
import numpy as np

def quat_matrix(q):
    w,x,y,z=np.asarray(q,float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def transform(pos,quat):
    t=np.eye(4);t[:3,:3]=quat_matrix(quat);t[:3,3]=pos;return t

def camera_poses(calibration,q):
    result=[]
    for cam in calibration['cameras']:
        t=np.eye(4)
        for body in cam['chain']:
            t=t@transform(body['pos'],body['quat'])
            for j in body['joints']:
                angle=float(q[j['encoder_index']])-j['reference_rad']
                axis=np.asarray(j['axis'],float);axis/=np.linalg.norm(axis)
                pivot=np.asarray(j['pos']);rotation=quat_matrix(np.r_[np.cos(angle/2),axis*np.sin(angle/2)])
                h=np.eye(4);h[:3,:3]=rotation;h[:3,3]=pivot-rotation@pivot;t=t@h
        result.append(t@transform(cam['pos'],cam['quat']))
    return result

def ray(center,fovy):
    focal=36/np.tan(np.deg2rad(fovy)/2) # 128px square, centered 128x72 content
    xy=np.asarray(center)*127
    return np.array([(xy[0]-63.5)/focal,-(xy[1]-63.5)/focal,-1.])

def epipolar_bbox_error(front,wrist,calibration,q):
    f,w=camera_poses(calibration,q)
    n=np.cross(f[:3,3]-w[:3,3],f[:3,:3]@ray(front['center'],calibration['cameras'][0]['fovy']))
    n=w[:3,:3].T@n
    focal=36/np.tan(np.deg2rad(calibration['cameras'][1]['fovy'])/2)
    line=np.array([n[0]/focal,-n[1]/focal,-n[2]-n[0]*63.5/focal+n[1]*63.5/focal])
    norm=np.linalg.norm(line[:2])
    if norm<1e-12:return float('inf')
    a,b,c,d=wrist['bbox'];values=np.array([line@np.array([x,y,1]) for x in [a,c] for y in [b,d]])/norm
    return 0. if values.min()<=0<=values.max() else float(np.min(np.abs(values)))

def bbox_match(a,b,pad):
    aa=a['bbox'];bb=b['bbox']
    return max(aa[0]-pad,bb[0])<=min(aa[2]+pad,bb[2]) and max(aa[1]-pad,bb[1])<=min(aa[3]+pad,bb[3])

class PublicAssociation:
    def __init__(self,rules):
        self.r=rules;self.front=None;self.wrist=None;self.front_t=None;self.wrist_t=None
    def update(self,fronts,wrists,q,t):
        conflict=False;front=None;wrist=None;reconfirmed=False
        if self.front is None:
            if fronts and (len(fronts)==1 or fronts[0]['area']>fronts[1]['area']):front=fronts[0]
        else:
            possible=[c for c in fronts if bbox_match(self.front,c,self.r['regional_match_padding_pixels'])]
            conflict=len(possible)>1
            if len(possible)==1 and t-self.front_t<=self.r['front_reconfirmation_max_seconds']+1e-9:
                front=possible[0];reconfirmed=t-self.front_t>.040000001
            # Never discard the old target and take a new largest red component.
        if front is not None:self.front=front;self.front_t=t
        if self.wrist is not None:
            possible=[c for c in wrists if bbox_match(self.wrist,c,self.r['regional_match_padding_pixels'])]
            conflict=conflict or len(possible)>1
            if len(possible)==1 and t-self.wrist_t<=self.r['maximum_unconfirmed_seconds']+1e-9:wrist=possible[0]
        elif front is not None:
            # Camera handover requires actual simultaneous geometrical evidence.
            possible=[c for c in wrists if epipolar_bbox_error(front,c,self.r['public_camera_calibration'],q)<=self.r['epipolar_pixel_tolerance']]
            conflict=conflict or len(possible)>1
            if len(possible)==1:wrist=possible[0]
        if front is not None and wrist is not None:
            conflict=conflict or epipolar_bbox_error(front,wrist,self.r['public_camera_calibration'],q)>self.r['epipolar_pixel_tolerance']
        if conflict:return None,None,True,False
        if wrist is not None:self.wrist=wrist;self.wrist_t=t
        return front,wrist,False,reconfirmed

class ReadinessEvidence:
    def __init__(self,rules):
        self.r=rules;self.intervals=[];self.previous_t=None;self.previous_kind=None;self.unknown_since=None
    def update(self,t,kind):
        if kind=='contradiction':self.intervals=[];self.unknown_since=None
        elif kind=='uncertain':
            if self.unknown_since is None:self.unknown_since=t
            if t-self.unknown_since>self.r['evidence_unknown_max_seconds']+1e-9:self.intervals=[]
        else:
            if self.unknown_since is not None and t-self.unknown_since>self.r['evidence_unknown_max_seconds']+1e-9:self.intervals=[]
            if self.previous_kind=='positive' and self.previous_t is not None and t>self.previous_t:
                self.intervals.append([self.previous_t,t])
            self.unknown_since=None
        cutoff=t-self.r['evidence_total_age_max_seconds']
        self.intervals=[[max(a,cutoff),b] for a,b in self.intervals if b>cutoff]
        self.previous_t=t;self.previous_kind=kind
        amount=sum(b-a for a,b in self.intervals)
        return {'kind':kind,'effective_positive_seconds':amount,'current_positive':kind=='positive',
                'unknown_since_s':self.unknown_since,'evidence_intervals_s':[x.copy() for x in self.intervals],
                'ready':kind=='positive' and amount>=self.r['start_evidence_seconds']-1e-9,
                'meaning':'finite-window effective readiness accumulation; not continuous readiness'}
