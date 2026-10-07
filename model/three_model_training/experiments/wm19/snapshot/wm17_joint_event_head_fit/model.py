"""Public local-history event residual; frozen WM14 outputs at zero final projection."""
from pathlib import Path
import importlib.util
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('wm15_frozen_WM14',ROOT.parent/'wm14_lateral_adaptation_fit/model.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SCALES=base.SCALES

class EntityMotionWM(base.EntityMotionWM):
    def __init__(self):
        super().__init__()
        self.local_action_context=nn.Sequential(nn.Linear(77,32),nn.GELU(),nn.Linear(32,32))
        self.local_spatial=nn.Sequential(nn.Conv2d(162,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GELU())
        self.local_attention=nn.Conv2d(64,29,1,bias=False)
        self.local_projection=nn.Conv2d(64,29,1)
        nn.init.zeros_(self.local_projection.weight);nn.init.zeros_(self.local_projection.bias)
        grid=torch.linspace(-1,1,16);yy,xx=torch.meshgrid(grid,grid,indexing='ij')
        self.register_buffer('local_coordinates',torch.stack((xx,yy))[None])

    def forward(self,history_images,history_state,history_valid,actions):
        captured={}
        hooks=[self.encoder.register_forward_hook(lambda m,a,v:captured.__setitem__('features',v)),
               self.spatial_history.register_forward_hook(lambda m,a,v:captured.__setitem__('history',v)),
               self.events.register_forward_hook(lambda m,a,v:captured.__setitem__('raw',v))]
        try:
            out=super().forward(history_images,history_state,history_valid,actions)
        finally:
            for hook in hooks:hook.remove()
        batch=history_images.shape[0]
        current=captured['features'].reshape(batch,3,64,16,16)[:,-1]
        available=(history_valid[:,:2].sum(1)>0).to(current.dtype)[:,None,None,None]
        historical=captured['history']*available
        state=history_state[:,-1]
        norm=state.new_tensor([3.]*6+[.08]+[1.]*6+[.08]+[3.]*6+[.08])
        action_scale=state.new_tensor([.12]*6+[.015])
        relative=(actions-state[:,None,:7])/action_scale
        context=self.local_action_context(torch.cat((state/norm,relative.flatten(1)),1))
        pixels=torch.cat((current,historical,context[:,:,None,None].expand(-1,-1,16,16),self.local_coordinates.expand(batch,-1,-1,-1)),1)
        local=self.local_spatial(pixels)
        attention=self.local_attention(local).flatten(2).softmax(2).reshape(batch,29,16,16)
        residual=(self.local_projection(local)*attention).sum((2,3))
        raw=captured['raw']+residual;total=raw[:,:9]
        def subset(c):return total+c-torch.logsumexp(torch.stack((torch.zeros_like(total),total,c)),dim=0)
        out['events']=torch.cat((total,subset(raw[:,9:18]),subset(raw[:,18:27]),raw[:,27:]),1)
        # Learned pooling is not a calibrated visibility or contact-location estimator.
        out['local_event_residual']=residual
        out['local_event_attention']=attention
        return out
