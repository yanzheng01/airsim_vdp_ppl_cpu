import torch    
import pyro
from torch import nn    
import pyro.distributions as dist
import torch.nn.functional as F
from cfg import *

dist.enable_validation(False)

rgb_stem = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1, bias=False),  # 3, 48, 64 -> 32, 48, 64
    nn.BatchNorm2d(32),  # 添加BatchNorm
    nn.MaxPool2d(2, 2),  # 32, 48, 64 -> 32, 24, 32
    nn.LeakyReLU(0.05),
    nn.Conv2d(32, 64, 3, padding=1, bias=False),  # 32, 24, 32 -> 64, 24, 32
    nn.BatchNorm2d(64),  # 添加BatchNorm
    nn.MaxPool2d(2, 2),  # 64, 24, 32 -> 64, 12, 16
    nn.LeakyReLU(0.05),
    nn.Conv2d(64, 128, 3, padding=1, bias=False),  # 64, 12, 16 -> 128, 12, 16
    nn.BatchNorm2d(128),  # 添加BatchNorm
    nn.MaxPool2d(2, 2),  # 128, 12, 16 -> 128, 6, 8
    nn.LeakyReLU(0.05),
    nn.Flatten(),
    nn.Linear(128*6*8, 768, bias=False),  # 128*6*8 = 6144
    nn.GroupNorm(1, 768),
    # nn.BatchNorm1d(768),  # 添加BatchNorm
    nn.LeakyReLU(0.05),
    nn.Linear(768, 192, bias=False),
    nn.GroupNorm(1, 192)
    # nn.BatchNorm1d(192),  # 添加BatchNorm
)

class Decoder(nn.Module):    
    def __init__(self, z_dim, rgb_stem, hidden_dim):
        super().__init__()
        self.z_dim = z_dim    
        self.rgb_stem = rgb_stem
        self.softplus = nn.Softplus()  
        self.sigmoid = nn.Sigmoid() 
        self.fc1 = nn.Linear(z_dim + 192, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16*12)    
        
    def forward(self, z, rgb):    
        rgb_x = self.rgb_stem(rgb)
        input = torch.cat([z, rgb_x], dim=1)   
        hidden = self.softplus(self.fc1(input))    
        img_param = self.sigmoid(self.fc2(hidden))  
        #img_param = img_param.view(-1, 1, 12, 16)  
        return img_param
    

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, 192)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.act = nn.LeakyReLU(0.05)


    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, None, hx
