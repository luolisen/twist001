"""Public three-frame prototype; current WM7 weights remain frozen."""
from pathlib import Path
import importlib.util
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_wm7_temporal_base',ROOT.parent/'wm_consistent_v7/model.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SCALES=base.SCALES

class TemporalWM(base.PhysicalWM):
    def __init__(self):
        super().__init__()
        self.history_fusion=nn.Sequential(nn.Linear(2*128+2*21+3,96),nn.ReLU(),nn.Linear(96,128))
        nn.init.normal_(self.history_fusion[-1].weight,std=.01);nn.init.zeros_(self.history_fusion[-1].bias)

    def forward(self,images,state,valid,actions):
        batch=images.shape[0]
        assert images.shape[1:]==(3,2,128,128,3) and state.shape[1:]==(3,21) and valid.shape[1:]==(3,)
        if not bool(torch.all(valid[:,-1]==1)):raise ValueError('Current observation must be valid')
        if not bool(torch.all((valid==0)|(valid==1))):raise ValueError('History validity must be binary')
        x=images.float().permute(0,1,2,5,3,4).reshape(batch*3,6,128,128)/255
        feature=self.encoder(x).reshape(batch,3,64,16,16)
        summary=self.image_summary(feature.flatten(0,1)).reshape(batch,3,128)
        norm=torch.tensor([3.]*6+[.08]+[1.]*6+[.08]+[3.]*6+[.08],device=state.device)
        scales=torch.tensor([.12]*6+[.015],device=state.device)
        relative=(actions-state[:,-1,None,:7])/scales
        h=self.trunk(torch.cat((summary[:,-1],state[:,-1]/norm,self.action_encoder(relative.flatten(1))),1))
        mask=valid[:,:2].float()
        image_delta=(summary[:,-1,None]-summary[:,:2])*mask[:,:,None]
        motor_delta=(state[:,-1,None]-state[:,:2])/norm*mask[:,:,None]
        history=torch.cat((image_delta.flatten(1),motor_delta.flatten(1),valid.float()),1)
        h=h+self.history_fusion(history)
        spatial=self.decoder(torch.cat((feature[:,-1],self.condition(h)[:,:,None,None].expand(-1,-1,16,16)),1))
        raw=self.events(h);total=raw[:,:9]
        def subset(conditional):return total+conditional-torch.logsumexp(torch.stack((torch.zeros_like(total),total,conditional)),dim=0)
        events=torch.cat((total,subset(raw[:,9:18]),subset(raw[:,18:27]),raw[:,27:]),1)
        return dict(motion=self.motion(h)*SCALES.to(state.device),events=events,regions=spatial[:,:18].reshape(batch,2,9,32,32),grasp_region=spatial[:,18:20],future_rgb32=spatial[:,20:].sigmoid().reshape(batch,2,3,32,32))
