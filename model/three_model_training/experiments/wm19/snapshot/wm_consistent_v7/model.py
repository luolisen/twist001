"""Physical forecasts from public dual RGB/motor state/actions only."""
import torch
from torch import nn
from torch.nn import functional as F
SCALES=torch.tensor([.12]*6+[.015]+[.05]*3)
class PhysicalWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(6,24,5,2,2),nn.ReLU(),nn.Conv2d(24,48,3,2,1),nn.ReLU(),nn.Conv2d(48,64,3,2,1),nn.ReLU())
        self.image_summary=nn.Sequential(nn.Flatten(),nn.Linear(64*16*16,128),nn.LayerNorm(128),nn.ReLU())
        self.action_encoder=nn.Sequential(nn.Linear(56,96),nn.ReLU(),nn.Linear(96,64),nn.ReLU())
        self.trunk=nn.Sequential(nn.Linear(128+21+64,192),nn.ReLU(),nn.Linear(192,128),nn.ReLU())
        self.motion=nn.Linear(128,10);self.events=nn.Linear(128,29)
        self.condition=nn.Linear(128,32)
        self.decoder=nn.Sequential(nn.Conv2d(96,64,3,padding=1),nn.ReLU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,26,3,padding=1))
    def forward(self,images,state,actions):
        b=images.shape[0];x=images.float().permute(0,1,4,2,3).reshape(b,6,128,128)/255
        feature=self.encoder(x)
        norm=torch.tensor([3.]*6+[.08]+[1.]*6+[.08]+[3.]*6+[.08],device=state.device)
        scale=torch.tensor([.12]*6+[.015],device=state.device)
        relative=(actions-state[:,None,:7])/scale
        h=self.trunk(torch.cat([self.image_summary(feature),state/norm,self.action_encoder(relative.flatten(1))],1))
        spatial=self.decoder(torch.cat([feature,self.condition(h)[:,:,None,None].expand(-1,-1,16,16)],1))
        raw=self.events(h)
        total=raw[:,:9]
        def subset(conditional):
            return total+conditional-torch.logsumexp(torch.stack((torch.zeros_like(total),total,conditional)),dim=0)
        events=torch.cat((total,subset(raw[:,9:18]),subset(raw[:,18:27]),raw[:,27:]),dim=1)
        return {'motion':self.motion(h)*SCALES.to(state.device),'events':events,
                'regions':spatial[:,:18].reshape(b,2,9,32,32),
                'grasp_region':spatial[:,18:20],
                'future_rgb32':spatial[:,20:].sigmoid().reshape(b,2,3,32,32)}
