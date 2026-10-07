"""Frozen WM9 outputs plus distinct public entity-motion supervision head."""
from pathlib import Path
import importlib.util
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_wm8_spatial_base',ROOT.parent/'wm_temporal_fit_v8/model.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SCALES=base.SCALES

class EntityMotionWM(base.TemporalWM):
    def __init__(self):
        super().__init__()
        self.spatial_history=nn.Sequential(nn.Conv2d(192,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU())
        self.public_history_gate=nn.Conv2d(64,1,1)
        self.spatial_residual=nn.Conv2d(64,64,1)
        self.history_readout=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,128))
        self.observed_flow_head=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,4,3,padding=1))
        self.entity_flow_head=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,4,3,padding=1))
        for layer in (self.spatial_residual,self.history_readout[-1]):
            nn.init.normal_(layer.weight,std=.001);nn.init.zeros_(layer.bias)

    def forward(self,images,state,valid,actions):
        batch=images.shape[0]
        assert images.shape[1:]==(3,2,128,128,3) and state.shape[1:]==(3,21) and valid.shape[1:]==(3,)
        if not bool(torch.all(valid[:,-1]==1)):raise ValueError('Current observation must be valid')
        if not bool(torch.all((valid==0)|(valid==1))):raise ValueError('History validity must be binary')
        # Replace missing frames BEFORE encoding/arithmetic. Even NaN junk cannot leak.
        safe_images=torch.where(valid[:,:,None,None,None,None].bool(),images,images[:,-1,None].expand_as(images))
        safe_state=torch.where(valid[:,:,None].bool(),state,state[:,-1,None].expand_as(state))
        x=safe_images.float().permute(0,1,2,5,3,4).reshape(batch*3,6,128,128)/255
        feature=self.encoder(x).reshape(batch,3,64,16,16)
        summary=self.image_summary(feature.flatten(0,1)).reshape(batch,3,128)
        norm=torch.tensor([3.]*6+[.08]+[1.]*6+[.08]+[3.]*6+[.08],device=state.device)
        scales=torch.tensor([.12]*6+[.015],device=state.device)
        relative=(actions-safe_state[:,-1,None,:7])/scales
        h=self.trunk(torch.cat((summary[:,-1],safe_state[:,-1]/norm,self.action_encoder(relative.flatten(1))),1))
        mask=valid[:,:2].float()
        image_delta=(summary[:,-1,None]-summary[:,:2])*mask[:,:,None]
        motor_delta=(safe_state[:,-1,None]-safe_state[:,:2])/norm*mask[:,:,None]
        h=h+self.history_fusion(torch.cat((image_delta.flatten(1),motor_delta.flatten(1),valid.float()),1))
        delta=(feature[:,-1,None]-feature[:,:2])*mask[:,:,None,None,None]
        z=self.spatial_history(torch.cat((feature[:,-1],delta.flatten(1,2)),1))
        available=(mask.sum(1)>0).float()[:,None,None,None]
        gate=self.public_history_gate(z).sigmoid()*available
        gated=z*gate
        h=h+self.history_readout(gated)*available.flatten(1)
        current=feature[:,-1]+self.spatial_residual(gated)*available
        spatial=self.decoder(torch.cat((current,self.condition(h)[:,:,None,None].expand(-1,-1,16,16)),1))
        raw=self.events(h);total=raw[:,:9]
        def subset(c):return total+c-torch.logsumexp(torch.stack((torch.zeros_like(total),total,c)),dim=0)
        events=torch.cat((total,subset(raw[:,9:18]),subset(raw[:,18:27]),raw[:,27:]),1)
        flow=self.observed_flow_head(gated).reshape(batch,2,2,32,32)*valid[:,1,None,None,None,None].float()
        entity_flow=self.entity_flow_head(gated).reshape(batch,2,2,32,32)*valid[:,1,None,None,None,None].float()
        return dict(entity_flow=entity_flow,motion=self.motion(h)*SCALES.to(state.device),events=events,regions=spatial[:,:18].reshape(batch,2,9,32,32),grasp_region=spatial[:,18:20],future_rgb32=spatial[:,20:].sigmoid().reshape(batch,2,3,32,32),observed_flow=flow,public_history_gate=gate)
