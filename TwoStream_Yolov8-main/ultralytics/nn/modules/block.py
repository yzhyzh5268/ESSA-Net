# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from torch.nn.init import trunc_normal_
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "Concat2",
    "ADD",
    "SimAM",
    "ShuffleAttention",
    "GAM_Attention",
    "CBAM2",
    "CoordAtt",
    "ECA",
    "SEAttention",
    "GLCBAM",
    "S2Attention",
    "SKAttention",
    "GLF",
    "NAM",
    "GCBAM",
    "SACBAM",
    "MdC2f",
    "C2f_Invo",
    "CAFFBlock",
    "CBAMk",
    "GateBlock",
    "ELAN",
    "C3k",
    "C3k2",
    "ESSA"
)
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.SiLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, n=3,stride=1, scale=0.1, map_reduce=4):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=1),
            BasicConv(2*inter_planes, 2*inter_planes , kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2*inter_planes, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
        )



    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out
    
class C2f_FEM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList([*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n//2)),FEM(self.c,self.c)] )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
import numpy as np
import torch
from torch import nn
from torch.nn import init

# https://arxiv.org/abs/2108.01072
def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x


def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=3):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) 
        a=torch.sum(torch.sum(x_all,1),1) 
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) 
        hat_a=hat_a.reshape(b,self.k,c) 
        bar_a=self.softmax(hat_a) 
        attention=bar_a.unsqueeze(-2) 
        out=attention*x_all 
        out=torch.sum(out,1).reshape(b,h,w,c)
        return out
#NAM
class NAM(nn.Module):
    def __init__(self, channels,c2, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.conv=Conv(channels,c2,1,1)
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
 
    def forward(self, x):
        x=torch.cat(x,1)
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        x=self.conv(x)
        return x
    
    
class GLF(nn.Module):

    def __init__(self, c1,c2,channel=512, reduction=16):
        super().__init__()
        channel=c1
        self.conv=Conv(c1,c2,1,1)
        self.d=1

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #全局池化
        # 全局特征提取
        self.fc1 = nn.Sequential(
         
            nn.Conv2d(channel, channel // reduction,1,1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,1,1),
            nn.BatchNorm2d(channel ),
            nn.Sigmoid()
        )
        # 局部特征提取
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,1,1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,1,1),
            nn.BatchNorm2d(channel),
        )



    def forward(self, x):
        x=torch.cat(x, self.d)
        b, c, _, _ = x.size()
        
        # 全局特征mul
        y = self.avg_pool(x)
        y = self.fc1(y).view(b, c, 1, 1)

        #局部特征
        y1= self.fc2(x)

        x=x * y.expand_as(x) 
        #局部特征add
        x=torch.add(x, y1)
        
        x=self.conv(x)

        return x
    




from collections import OrderedDict


class SKAttention(nn.Module):

    def __init__(self,c1,c2, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):

        x=torch.cat(x,1)
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        V=self.conv(V)
        return V


    


class S2Attention(nn.Module):

    def __init__(self, c1,c2,channels=512 ):
        super().__init__()
        channels=c1
        self.conv=Conv(c1,c2,1,1)

        self.mlp1 = nn.Linear(channels,channels*3)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention(c1)

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b,c,w,h = x.size()
        x=x.permute(0,2,3,1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all=torch.stack([x1,x2,x3],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x=x.permute(0,3,1,2)
        x=self.conv(x)
        return x
  
  

 
 
###################### EffectiveSE     ####     end   by  AI&CV  ###############################

import numpy as np
import torch
from torch import nn
from torch.nn import init

class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        #self.act=nn.SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out

class CBAM2(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM2, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x=torch.cat(x, self.d) 
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        x=self.conv(out)
        return x


class CSFM(nn.Module):
    def __init__(self, c1,c2):
        super(CSFM, self).__init__()
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        _,c,_,_=x[0].shape
        x3=x[0]
        x4=x[1]
        x=torch.cat(x, self.d) 
        out = self.channel_attention(x) * x
        x1, x2 = torch.split(out, c, dim =self.d)

        x1=x1*x3
        x2=x2*x4
        # x1+=x[0]
        # x2+=x[1]
        out=torch.add(x1,x2)
        # out = self.spatial_attention(out) * out
        
        return out

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output
    



class SACBAM(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        


        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)

        x = self.channel_shuffle(x, 2)
        x_channel=self.channel_attention(x) * x
        out=self.spatial_attention(x_channel) * x_channel
        out=self.conv(out)
        return out
    


    
class GCBAM(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        


        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b, c, h, w = x.size()
        # group into subfeatures

        # x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        # x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # # channel attention
        # x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        # x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        # x_channel = x_0 * self.sigmoid(x_channel)

        # # spatial attention
        # x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        # x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        # x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        x_channel=self.channel_attention(x) * x
        
        out=self.spatial_attention(x_channel) * x_channel
        # concatenate along channel axis
        # out = torch.cat([x_channel, x_spatial], dim=1) 
        # out = out.contiguous().view(b, -1, h, w)
        # channel shuffle
        out = self.channel_shuffle(out, 2)
        out=self.conv(out)
        return out
    
    
# 局部CBAM
class GLCBAM(nn.Module):
    def __init__(self, c1,c2):
        super(GLCBAM, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()
        mid_channel=c1//16
        
        #局部特征
        self.localConv = nn.Sequential(          
            nn.Conv2d(in_channels=c1, out_channels=mid_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=c1,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(c1),
        )

    def forward(self, x):
        x=torch.cat(x, self.d) 
        y=x
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        
        local=self.localConv(y)
        out=torch.add(local,out)
        
        x=self.conv(out)

        return x


class SACBAM(nn.Module):
    def __init__(self, c1,c2):
        super(SACBAM, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()
        self.SA=ShuffleAttention(c1,c2)
        mid_channel=c1//16
        
 

    def forward(self, x):
        x=torch.cat(x, self.d) 
        y=x
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        
        local=self.SA(y)
        out=torch.add(local,out)
    
        x=self.conv(out)

        return x
    

class SEAttention(nn.Module):

    def __init__(self, c1,c2,channel=512, reduction=16):
        super().__init__()
        channel=c1
        # self.conv=Conv(c1,c2,1,1)
        self.d=1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x=x * y.expand_as(x) 
        # x=self.conv(x)
        return x
    
class Concat2(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1,c2,dimension=1):
        super().__init__()
        self.d = dimension#沿着哪个维度进行拼接
        #self.conv=nn.Conv2d(c1,c2,1,1,bias=False)
        self.conv=Conv(c1,c2,1,1)

    def forward(self, x):
        x=torch.cat(x, self.d)
        x=self.conv(x)
        return x
class SA(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out
    
from torch.nn import init
from torch.nn.parameter import Parameter

class SimAM(torch.nn.Module):
    def __init__(self, c1,c2,e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.d=1
        self.conv=Conv(c1,c2,1,1)

        

    def forward(self, x):
        x=torch.cat(x, self.d)
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )
        x= x * self.activaton(y)
        x=self.conv(x)
        return x




import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
class CoordAtt(nn.Module):
    def __init__(self, inp,c2, reduction=32):
        super(CoordAtt, self).__init__()
        self.conv=Conv(inp,c2,1,1)
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction)
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
 
    def forward(self, x):
        x=torch.cat(x,dim=1)
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return self.conv(out)

import torch
from torch import nn
from torch.nn.parameter import Parameter
class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x
    
# class ECA(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, c1,c2, k_size=3):
#         super(ECA, self).__init__()
#         self.conv1=Conv(c1,c2,1,1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         x=torch.cat(x,dim=1)
#         y = self.avg_pool(x)
 
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
 
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
 
#         return self.conv1(x * y.expand_as(x))
    
class GAM_Attention(nn.Module):
    # https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()
        self.conv=Conv(c1,c2,1,1)

        c2=c1
        self.d=1
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        x=torch.cat(x,dim=self.d)
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        # out=channel_shuffle(out,4) #last shuffle
        out=self.conv(out)
        return out
    
class ShuffleAttention(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16, G=8):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        out=self.conv(out)
        return out






        
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        # a 8400
        # c1=16 
        # 4 16 a 
        # 16 4 a 
        # 4 a
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class MdC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Md(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0,deiltations=i+1) for i in range(n))
        

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class CDC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        # 3 1 3 8 5 2  5 2     k= 3, 3, 5, and 5 and d= 1, 8, 2, and 3
        if n==1:
           # high pass d
           # 3 1 /3 8/5 3
           self.m = nn.ModuleList(Md(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0,deiltations=8))


        else :
           # low pass c
           # 3 1/3 8/ 5 2/ 5 3/ 3 3/ 5 5 
           self.m = nn.ModuleList((Md(self.c, self.c, shortcut, g, k=((3, 3), (5, 5)), e=1.0,deiltations=2),
                                  Md(self.c, self.c, shortcut, g, k=((3, 3), (5, 5)), e=1.0,deiltations=3)) )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    


class C2f_F(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = c1//4 # hidden channels
        self.c1=self.c*3
        self.cv1 = Conv(c1, c1, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c,self.c,k=3,s=1) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x=self.cv1(x)
        c1=3*self.c
        x1 = x[:, :c1, :, :]  
        # 第二部分  
        x2 = x[:, c1:, :, :] 
        y=list([x1,x2])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


import torch
import torch.nn as nn
from torch.nn import functional as F
 
 

 
class Involution(nn.Module):
 
    def __init__(self, c1, c2, kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c1
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.c1 // self.group_channels
        self.conv1 = Conv(
            c1, c1 // reduction_ratio, 1)
        self.conv2 = Conv(
            c1 // reduction_ratio,
            kernel_size ** 2 * self.groups,
            1, 1)
 
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)
 
    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.c1, h, w)
 
        return out

from ultralytics.utils.torch_utils import make_divisible


class PKIModule_CAA(nn.Module):
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor
    

class PKIModule(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)
        
        self.pre_conv = Conv(inc, hidc)
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)
        self.post_conv = Conv(hidc, ouc)
        
        if with_caa:
            self.caa_factor = PKIModule_CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None
        
        self.add_identity = add_identity and inc == ouc
    
    def forward(self, x):
        x = self.pre_conv(x)
        
        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        x = self.pw_conv(x)
        
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x
    


class C2f_PKIModule(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        self.m = nn.ModuleList(PKIModule(self.c, self.c, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n))

class ShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride):  # ch_in, ch_out, stride
        super().__init__()

        self.stride = stride

        branch_features = oup // 2 # 输出的一半
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride == 2:
            # copy input
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride == 2) else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            #Dw卷积
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            nn.BatchNorm2d(branch_features),
            #Pw
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = self.channel_shuffle(out, 2)

        return out

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

        return out
    
class C2f_Shufflenet(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ShuffleNetV2(self.c, self.c,1) for _ in range(n))

class C2f_Invo(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(InvoConv(self.c, self.c,1) for _ in range(n))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class InvoConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Involution(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class Md(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,deiltations=1):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g,d=deiltations)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
 #寻找add   
class ADD(nn.Module):
    #  Add two tensors
    
    def __init__(self, arg):
        super(ADD,self).__init__()
        # 128 256 512
        self.arg = arg
  
    def forward(self, x):
        return torch.add(x[0], x[1])



class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

# https://mp.weixin.qq.com/s/26H0PgN5sikD1MoSkIBJzg
class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = Conv(in_features, filters, act=False)
         self.c1 = Conv(filters, filters, 3)
         self.c2 = Conv(filters, filters, 3)
         self.c3 = Conv(filters, filters, 3)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.drop = nn.Dropout2d(0.1)
         self.bn1 = nn.BatchNorm2d(filters)
         self.silu = nn.SiLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x


class C2f_PPA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(PPA(self.c, self.c) for _ in range(n))

from timm.models.layers import DropPath


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        # x = x.clone()  # !!! Keep the original input intact for the residual connection later
        # x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        # return x
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=1,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            # nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        # self.adjust_channel = None
        # if inc != dim:
        #     self.adjust_channel = Conv(inc, dim, 1)

        # if layer_scale_init_value > 0:
        #     self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        #     self.forward = self.forward_layer_scale
        # else:
        #     self.forward = self.forward

    def forward(self, x):
        # if self.adjust_channel is not None:
        #     x = self.adjust_channel(x)
        # shortcut = x
        x = self.spatial_mixing(x)
        # x = shortcut + self.drop_path(self.mlp(x))
        
        return self.mlp(x)

    def forward_layer_scale(self, x):
        # shortcut = x
        x = self.spatial_mixing(x)
        # x = shortcut + self.drop_path(
        #     self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C2f_Faster(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))















class RepGhostModule(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True,
            reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.SiLU(inplace=False)
        else:
            self.relu = nn.Sequential()
    

    def forward(self, x):
        

        x1 = self.primary_conv(x)  # mg
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn   # 激活函数
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
 
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)



class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )
          

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x) #
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck mg
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)
    

class RepGhostModule(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True,
            reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy
        # 1x1 conv + bn + SiLU
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv) #indentity
        self.fusion_bn = nn.Sequential(*fusion_bn) #fusion bn

        # dwconv BN Silu
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.SiLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)  # conv1x1 SiLu
        x2 = self.cheap_operation(x1) # dw BN SiLu
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))# indentity x1 + bn
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        mid_chs=in_chs//2
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck mg
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)

class C2f_RG(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepGhostBottleneck(self.c, self.c) for _ in range(n))




### bifpn##


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU()(self.shuf(x2))

class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)

class GSBottleneckns(GSBottleneck):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__(c1, c2, k, s, e)
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConvns(c1, c_, 1, 1),
            GSConvns(c_, c2, 3, 1, act=False))
        
class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)

class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))

class VoVGSCSPns(VoVGSCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.gsb = nn.Sequential(*(GSBottleneckns(c_, c_, e=1.0) for _ in range(n)))

class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)


class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans
    







class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion
        
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
        
    
    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)
        
###### bifpn###


 

class Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()

        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input) ## 进行特征压缩 因为只计算一个特征的权重
        local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
        global_w = self.global_att(agg_input) ## 全局注意力 即channel attention
        w = self.sigmoid(local_w * global_w) ## 计算特征x1的权重
        xo = w * x1 + (1 - w) * x2 ## fusion results ## 特征聚合
        return xo
class Concat3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1,c2,dimension=1):
        super().__init__()
        self.d = dimension#沿着哪个维度进行拼接
        self.Fm=Fusion_module(channels=c2)


    def forward(self, x):
        # x1=self.conv1(x[0])
        # x2=self.conv2(x[1])

        x=self.Fm(x[0],x[1])
        # x=torch.cat([x1,x2], self.d)

        return x
################空###################

# class RIFusion(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, c1,r=16,dimension=1):
#         super().__init__()

#     def forward(self, x):
#         return x

# class RIFusion(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, c1,r=16,dimension=1):
#         super().__init__()
#         self.c1=c1*2
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(self.c1, self.c1 // r, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.c1 // r, self.c1, bias=False),
#             nn.Sigmoid()
#             # nn.Sigmoid(inplace=True)
#         )
#     def forward(self, x):
#         # return x
#         b, _, _, _ = x.size()
#         y = self.avg_pool(x).view(b, self.c1)
#         y = self.fc(y).view(b, self.c1, 1, 1)
  
#         x1=x*y
#         return x+torch.cat((x1[:,self.c1//2:,...],x1[:,:self.c1//2,...]),dim=1)

#效果最好hao RIFusion
class RIFusion(nn.Module):
    def __init__(self, c1, r=16, dimension=1):
        """
        c1: 单分支通道数（RGB or IR）
        r:  通道注意力的压缩比
        """
        super().__init__()
        self.c1 = c1 * 2  # 双流融合后通道数
        # 全局通道注意力 (SE block)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.c1, self.c1 // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c1 // r, self.c1, bias=False),
            nn.Sigmoid()
        )
        # 轻量通道交互 (1x1 Conv 替代简单切分)
        self.conv1x1 = nn.Conv2d(self.c1, self.c1, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.c1)
        self.act = nn.SiLU(inplace=True)  # 更稳定的激活函数

    def channel_shuffle(self, x, groups=2):
        """ShuffleNet风格的通道混合，增强RGB/IR交互"""
        b, c, h, w = x.size()
        x = x.view(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        b, _, _, _ = x.size()
        # 通道注意力
        y = self.avg_pool(x).view(b, self.c1)
        y = self.fc(y).view(b, self.c1, 1, 1)
        x = x * y.expand_as(x)

        # 通道交互 (1x1 conv + shuffle)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.channel_shuffle(x, groups=2)

        return x

# class RIFusion(nn.Module):
#     def __init__(self, c1, r=16, groups=2, bottleneck_factor=4):
#         super().__init__()
#         self.c_total = c1 * 2
#         assert self.c_total % groups == 0
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # conv-based fc
#         self.fc = nn.Sequential(
#             nn.Conv2d(self.c_total, self.c_total // r, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.c_total // r, self.c_total, 1, bias=False),
#             nn.Sigmoid()
#         )
#         # 使用分组 pointwise 减少参数
#         self.conv1x1 = nn.Conv2d(self.c_total, self.c_total, kernel_size=1, groups=groups, bias=False)
#         self.bn = nn.BatchNorm2d(self.c_total)
#         self.act = nn.SiLU(inplace=True)
#         self.groups = groups

#     def channel_shuffle(self, x):
#         b, c, h, w = x.size()
#         g = self.groups
#         x = x.view(b, g, c // g, h, w).permute(0, 2, 1, 3, 4).contiguous()
#         return x.view(b, c, h, w)

#     def forward(self, x):
#         assert x.size(1) == self.c_total
#         identity = x
#         y = self.fc(self.avg_pool(x))       # (B, C_total, 1, 1)
#         x = x * y
#         x = self.conv1x1(x)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.channel_shuffle(x)
#         return x + identity


# class RIFusion(nn.Module):
#     """RGB-IR特征融合模块，使用注意力机制"""
#     def __init__(self, input_channels, output_channels=None, reduction=16):
#         super(RIFusion, self).__init__()
#         if output_channels is None:
#             output_channels = input_channels
#         self.input_channels = input_channels
#         self.output_channels = output_channels
        
#         # 通道注意力 
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(input_channels, input_channels // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(input_channels // reduction, input_channels, 1),
#             nn.Sigmoid()
#         )
        
#         # 空间注意力
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=7, padding=3),
#             nn.Sigmoid()
#         )
        
#         # 特征融合
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(input_channels, output_channels, 1),
#             nn.BatchNorm2d(output_channels),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, x):
#         """
#         x: [rgb_features, ir_features] 
#         两个输入特征图的列表
#         """
#         if isinstance(x, list) and len(x) == 2:
#             rgb_feat, ir_feat = x
#         else:
#             raise ValueError("RIFusion需要两个输入特征图")
            
#         # 确保特征图尺寸一致
#         if rgb_feat.shape != ir_feat.shape:
#             # 调整IR特征图尺寸与RGB一致
#             ir_feat = nn.functional.interpolate(ir_feat, size=rgb_feat.shape[2:], mode='bilinear')
        
#         # 特征连接
#         concat_feat = torch.cat([rgb_feat, ir_feat], dim=1)
        
#         # 通道注意力
#         ca_weight = self.channel_attention(concat_feat)
#         ca_feat = concat_feat * ca_weight
        
#         # 分离RGB和IR通道
#         rgb_ca, ir_ca = torch.chunk(ca_feat, 2, dim=1)
        
#         # 空间注意力
#         avg_out = torch.mean(concat_feat, dim=1, keepdim=True)
#         max_out, _ = torch.max(concat_feat, dim=1, keepdim=True)
#         spatial_input = torch.cat([avg_out, max_out], dim=1)
#         sa_weight = self.spatial_attention(spatial_input)
        
#         # 应用空间注意力
#         rgb_sa = rgb_ca * sa_weight
#         ir_sa = ir_ca * sa_weight
        
#         # 最终融合
#         fused_feat = torch.cat([rgb_sa, ir_sa], dim=1)
#         output = self.fusion_conv(fused_feat)
        
#         return output


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, channels):
        super(MultiScaleFusion, self).__init__()
        self.channels = channels
        
        # 权重学习网络
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征调整
        self.feature_adjust = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        x: [fusion_feat, rgb_feat, ir_feat]
        三个特征图的列表
        """
        if isinstance(x, list) and len(x) == 3:
            fusion_feat, rgb_feat, ir_feat = x
        else:
            raise ValueError("MultiScaleFusion需要三个输入特征图")
            
        # 确保所有特征图尺寸一致
        target_size = fusion_feat.shape[2:]
        if rgb_feat.shape[2:] != target_size:
            rgb_feat = nn.functional.interpolate(rgb_feat, size=target_size, mode='bilinear')
        if ir_feat.shape[2:] != target_size:
            ir_feat = nn.functional.interpolate(ir_feat, size=target_size, mode='bilinear')
        
        # 连接所有特征
        concat_feat = torch.cat([fusion_feat, rgb_feat, ir_feat], dim=1)
        
        # 学习融合权重
        weights = self.weight_net(concat_feat)  # [B, 3, 1, 1]
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)
        
        # 加权融合
        fused = w1 * fusion_feat + w2 * rgb_feat + w3 * ir_feat
        
        # 特征调整
        output = self.feature_adjust(fused)
        
        return output


class BiFPN(nn.Module):
    """跨模态注意力模块（可选的增强版本）"""
    def __init__(self, channels, num_heads=8):
        super(BiFPN, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.q_conv = nn.Conv2d(channels, channels, 1)
        self.k_conv = nn.Conv2d(channels, channels, 1)
        self.v_conv = nn.Conv2d(channels, channels, 1)
        
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, rgb_feat, ir_feat):
        B, C, H, W = rgb_feat.shape
        
        # 生成查询、键、值
        q = self.q_conv(rgb_feat).view(B, self.num_heads, self.head_dim, H*W)
        k = self.k_conv(ir_feat).view(B, self.num_heads, self.head_dim, H*W)
        v = self.v_conv(ir_feat).view(B, self.num_heads, self.head_dim, H*W)
        
        # 注意力计算
        attn = (q.transpose(-2, -1) @ k) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        
        # 应用注意力
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)
        
        # 投影和残差连接
        out = self.proj(out) + rgb_feat
        
        return out
# # class RIFusion(nn.Module):
#     """
#     改进版本3: 多路径融合与残差学习
#     - 采用多路径设计增强特征表达
#     - 引入残差学习机制
#     - 使用自适应权重分配
#     """
#     def __init__(self, c1, r=16, dimension=1):
#         super().__init__()
#         self.c1 = c1 * 2
        
#         # 路径1: 全局注意力
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.global_fc = nn.Sequential(
#             nn.Linear(self.c1, self.c1 // r, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.c1 // r, self.c1, bias=False)
#         )
        
#         # 路径2: 局部注意力
#         self.local_conv = nn.Sequential(
#             nn.Conv2d(self.c1, self.c1 // r, 1, bias=False),
#             nn.BatchNorm2d(self.c1 // r),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.c1 // r, self.c1, 1, bias=False),
#             nn.BatchNorm2d(self.c1)
#         )
        
#         # 路径3: 空间注意力
#         self.spatial = nn.Sequential(
#             nn.Conv2d(self.c1, 1, 1, bias=False),
#             nn.Sigmoid()
#         )
        
#         # 自适应权重
#         self.weight_fc = nn.Sequential(
#             nn.Linear(self.c1, 3, bias=False),
#             nn.Softmax(dim=1)
#         )
        
#         # 输出投影
#         self.project = nn.Conv2d(self.c1, self.c1, 1, bias=False)
        
#     def forward(self, x):
#         b, c, h, w = x.size()
        
#         # 路径1: 全局通道注意力
#         global_feat = self.global_avg_pool(x).view(b, self.c1)
#         global_att = torch.sigmoid(self.global_fc(global_feat)).view(b, self.c1, 1, 1)
#         path1 = x * global_att
        
#         # 路径2: 局部通道注意力
#         local_att = torch.sigmoid(self.local_conv(x))
#         path2 = x * local_att
        
#         # 路径3: 空间注意力
#         spatial_att = self.spatial(x)
#         path3 = x * spatial_att
        
#         # 自适应权重计算
#         weights = self.weight_fc(global_feat)  # [b, 3]
        
#         # 加权融合
#         fusion = (path1 * weights[:, 0:1, None, None] + 
#                  path2 * weights[:, 1:2, None, None] + 
#                  path3 * weights[:, 2:3, None, None])
        
#         # 特征重组
#         rgb_feat = fusion[:, :self.c1//2, ...]
#         ir_feat = fusion[:, self.c1//2:, ...]
#         x_reorg = torch.cat([ir_feat, rgb_feat], dim=1)
        
#         # 投影并残差连接
#         out = self.project(x_reorg)
        
#         return x + out
# # class RIFusion(nn.Module):
#     """
#     双分支融合模块（兼容 Ultralytics parse_model & stride 探测）：
#     - __init__(c1, r=16, ...)：c1 = 单分支通道数
#     - forward 支持三种输入： (x_rgb, x_ir) 或 [x_rgb, x_ir] 或 single-tensor dummy (B,c1,H,W)
#     - 输出为单分支通道数 self.single_c（通过 1x1 conv 将 2*c -> c）
#     """
#     def __init__(self, c1: Optional[int]=None, r: int = 16, *args, **kwargs):
#         super().__init__()
#         # 尝试从位置参数或 kwargs 中解析 c1（单分支的通道数）
#         if c1 is None:
#             for a in args:
#                 if isinstance(a, int):
#                     c1 = a
#                     break
#         if c1 is None and 'c1' in kwargs:
#             c1 = kwargs.get('c1')

#         if c1 is None:
#             raise ValueError("RIFusion requires c1 (single-branch channels) as an int argument")

#         self.single_c = c1
#         self.c_concat = c1 * 2

#         # 通道注意力（基于 concat）
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(self.c_concat, max(1, self.c_concat // r), bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(max(1, self.c_concat // r), self.c_concat, bias=False),
#             nn.Sigmoid()
#         )

#         # 关键：把 2*c -> c 的降维 conv（保证下游收到期望的通道数）
#         self.reduce = nn.Sequential(
#             nn.Conv2d(self.c_concat, self.single_c, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(self.single_c),
#             nn.ReLU(inplace=True)
#         )

#     def _align_channels(self, x1, x2):
#         c1 = x1.shape[1]
#         c2 = x2.shape[1]
#         if c1 == c2:
#             return x1, x2
#         # 若其中一个是1通道（常见 IR 单通道），repeat 到另一分支通道数
#         if c1 == 1:
#             x1 = x1.repeat(1, c2, 1, 1)
#             return x1, x2
#         if c2 == 1:
#             x2 = x2.repeat(1, c1, 1, 1)
#             return x1, x2
#         # 其他不匹配情形，抛出错误（你也可以改成 1x1 conv 做映射）
#         raise RuntimeError(f"RIFusion channel mismatch and neither is 1-ch: {c1} vs {c2}")

#     def forward(self, x, x2=None):
#         # 支持 list/tuple 两路输入
#         if x2 is None:
#             if isinstance(x, (list, tuple)):
#                 if len(x) != 2:
#                     raise TypeError(f"RIFusion expects two tensors, got list len {len(x)}")
#                 x_rgb, x_ir = x[0], x[1]
#             elif torch.is_tensor(x):
#                 # 已 concat 的情况：通道==2*c -> 处理并降维
#                 if x.dim() == 4 and x.shape[1] == self.c_concat:
#                     cat = x
#                     # attention
#                     b = cat.size(0)
#                     y = self.avg_pool(cat).view(b, self.c_concat)
#                     y = self.fc(y).view(b, self.c_concat, 1, 1)
#                     x_att = cat * y
#                     # 你原来的交换/残差操作（保持），然后降维
#                     half = self.c_concat // 2
#                     fused = cat + torch.cat((x_att[:, half:, ...], x_att[:, :half, ...]), dim=1)
#                     out = self.reduce(fused)  # 降回 single_c
#                     return out
#                 # 单一路输入且通道==single_c（parse_model dummy forward），把它复制为两路相同输入
#                 if x.dim() == 4 and x.shape[1] == self.single_c:
#                     x_rgb = x
#                     x_ir = x.clone()
#                 else:
#                     raise TypeError(
#                         f"RIFusion single-tensor input should have {self.c_concat} channels for cat-mode or "
#                         f"{self.single_c} channels for dummy-mode, got {getattr(x,'shape',None)}")
#             else:
#                 raise TypeError(f"Unsupported input type for RIFusion.forward: {type(x)}")
#         else:
#             x_rgb, x_ir = x, x2

#         # 对齐通道（当 IR 是 1ch 时会 repeat）
#         x_rgb, x_ir = self._align_channels(x_rgb, x_ir)

#         # 拼接并做 channel-attention & 交叉残差（与你原逻辑一致）
#         cat = torch.cat((x_rgb, x_ir), dim=1)  # (B,2*c,H,W)
#         b = cat.size(0)
#         y = self.avg_pool(cat).view(b, self.c_concat)
#         y = self.fc(y).view(b, self.c_concat, 1, 1)
#         x_att = cat * y
#         half = self.c_concat // 2
#         fused = cat + torch.cat((x_att[:, half:, ...], x_att[:, :half, ...]), dim=1)

#         # 降维回单分支通道
#         out = self.reduce(fused)  # (B, single_c, H, W)
#         return out
# class RIFusion(nn.Module):
#     def __init__(self, c1: Optional[int]=None, r: int = 16, *args, **kwargs):
#         """
#         c1: single branch channel (single_c)
#         r: reduction ratio for channel-attention bottleneck
#         """
#         super().__init__()
#         # parse c1 from args/kwargs if needed
#         if c1 is None:
#             for a in args:
#                 if isinstance(a, int):
#                     c1 = a
#                     break
#         if c1 is None and 'c1' in kwargs:
#             c1 = kwargs.get('c1')
#         if c1 is None:
#             raise ValueError("RIFusion requires c1 (single-branch channels) as an int argument")
#         self.single_c = int(c1)
#         # c_concat is *expected* 2*channel_of_each_branch, but we don't require that here
#         self.r = max(1, int(r))
#         # channel attention on concatenated channels; we set fc based on dynamic c_concat at forward
#         # But to keep torchscript/nn.Module simple, we will build fc when we know c_concat for the first time
#         self._fc_built = False
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         # reduction conv: 2*c -> single_c  (but we will create when we know c_concat on first forward)
#         self.reduce = None

#         # placeholder for dynamically created convs to align mismatched channels (if neither side is 1-ch)
#         # these will be created lazily on first forward if needed and registered to the module.
#         self._align_created = False

#     def _build_attention_and_reduce(self, c_concat):
#         """Create fc and reduce layers once we know c_concat (2 * branch_channels)."""
#         if self._fc_built and hasattr(self, 'fc') and self.reduce is not None:
#             return
#         hidden = max(1, c_concat // self.r)
#         self.fc = nn.Sequential(
#             nn.Linear(c_concat, hidden, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, c_concat, bias=False),
#             nn.Sigmoid()
#         )
#         # reduction conv: map from c_concat -> single_c
#         self.reduce = nn.Sequential(
#             nn.Conv2d(c_concat, self.single_c, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(self.single_c),
#             nn.ReLU(inplace=True)
#         )
#         self._fc_built = True

#     def _align_channels(self, x1, x2):
#         c1 = x1.shape[1]
#         c2 = x2.shape[1]
#         # if equal -> OK
#         if c1 == c2:
#             return x1, x2
#         # if one is 1 -> repeat to match the other
#         if c1 == 1:
#             x1 = x1.repeat(1, c2, 1, 1)
#             return x1, x2
#         if c2 == 1:
#             x2 = x2.repeat(1, c1, 1, 1)
#             return x1, x2
#         # else: neither is 1 and channels mismatch. Create/attach 1x1 convs mapping both -> max(c1,c2) OR to self.single_c
#         # We'll map both to the same channel count = max(c1,c2) to avoid information loss, then proceed.
#         target = max(c1, c2)
#         if not self._align_created:
#             # create convs dynamically and register them so parameters are tracked
#             self.align_conv1 = nn.Conv2d(c1, target, kernel_size=1, bias=False)
#             self.align_bn1 = nn.BatchNorm2d(target)
#             self.align_conv2 = nn.Conv2d(c2, target, kernel_size=1, bias=False)
#             self.align_bn2 = nn.BatchNorm2d(target)
#             # register
#             self.add_module('align_conv1', self.align_conv1)
#             self.add_module('align_bn1', self.align_bn1)
#             self.add_module('align_conv2', self.align_conv2)
#             self.add_module('align_bn2', self.align_bn2)
#             self._align_created = True
#         # apply mapping
#         x1 = self.align_bn1(self.align_conv1(x1))
#         x2 = self.align_bn2(self.align_conv2(x2))
#         return x1, x2

#     def forward(self, x, x2=None):
#         # support list/tuple two-input form
#         if x2 is None:
#             if isinstance(x, (list, tuple)):
#                 if len(x) != 2:
#                     raise TypeError(f"RIFusion expects two tensors, got list len {len(x)}")
#                 x_rgb, x_ir = x[0], x[1]
#             elif torch.is_tensor(x):
#                 # already concatenated 2*c case
#                 if x.dim() == 4:
#                     cat = x
#                     c_concat = cat.shape[1]
#                     self._build_attention_and_reduce(c_concat)
#                     b = cat.size(0)
#                     y = self.avg_pool(cat).view(b, c_concat)
#                     y = self.fc(y).view(b, c_concat, 1, 1)
#                     x_att = cat * y
#                     half = c_concat // 2
#                     fused = cat + torch.cat((x_att[:, half:, ...], x_att[:, :half, ...]), dim=1)
#                     out = self.reduce(fused)
#                     return out
#                 else:
#                     raise TypeError(f"RIFusion single-tensor input should have 4 dims, got {x.dim()}")
#             else:
#                 raise TypeError(f"Unsupported input type for RIFusion.forward: {type(x)}")
#         else:
#             x_rgb, x_ir = x, x2

#         # align channels (handles 1-ch IR or general mismatch)
#         x_rgb, x_ir = self._align_channels(x_rgb, x_ir)

#         # concat and attention
#         cat = torch.cat((x_rgb, x_ir), dim=1)
#         c_concat = cat.shape[1]
#         self._build_attention_and_reduce(c_concat)
#         b = cat.size(0)
#         y = self.avg_pool(cat).view(b, c_concat)
#         y = self.fc(y).view(b, c_concat, 1, 1)
#         x_att = cat * y
#         half = c_concat // 2
#         fused = cat + torch.cat((x_att[:, half:, ...], x_att[:, :half, ...]), dim=1)

#         out = self.reduce(fused)  # (B, single_c, H, W)
#         return out


class MSCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 修复：确保groups数量正确
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        
        return attn * u

# 更安全的替代方案：自适应多尺度注意力模块
class AdaptiveMSCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用普通卷积避免groups问题
        self.conv1 = nn.Conv2d(dim, dim//4, 1)
        self.conv_3x3 = nn.Conv2d(dim//4, dim//4, 3, padding=1)
        self.conv_5x5 = nn.Conv2d(dim//4, dim//4, 5, padding=2)
        self.conv_7x7 = nn.Conv2d(dim//4, dim//4, 7, padding=3)
        self.conv_out = nn.Conv2d(dim//4*3, dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 降维
        x_reduced = self.conv1(x)
        
        # 多尺度特征提取
        x3 = self.conv_3x3(x_reduced)
        x5 = self.conv_5x5(x_reduced) 
        x7 = self.conv_7x7(x_reduced)
        
        # 特征融合
        multi_scale = torch.cat([x3, x5, x7], dim=1)
        attention = self.conv_out(multi_scale)
        attention = self.sigmoid(attention)
        
        return x * attention

# class RIFusion(nn.Module):
#     """安全版本，避免通道数不匹配问题"""
#     def __init__(self, c1, r=16, dimension=1):
#         super().__init__()
#         self.adaptive_msca = AdaptiveMSCA(c1)
        
#         # 通道注意力 - 使用自适应平均池化
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(c1, c1//r, 1)
#         self.fc2 = nn.Conv2d(c1//r, c1, 1)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         # 空间注意力
#         spatial_att = self.adaptive_msca(x)
        
#         # 通道注意力
#         channel_att = self.gap(x)
#         channel_att = F.relu(self.fc1(channel_att))
#         channel_att = self.sigmoid(self.fc2(channel_att))
        
#         # 融合输出
#         out = spatial_att * channel_att + x
#         return out




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class RIFusion(nn.Module):
#     """
#     改进且兼容的 RIFusion

#     保持签名：RIFusion(c1, r=16, dimension=1)
#     - 输入 x 形状 (B, 2*C, H, W)（或上游仍以单模态通道数 c1 调用）
#     - 模块会在第一次 forward 时自动推断半通道数（half_c）并完成内部模块构建（lazy init）。
#     - 主要步骤：
#         1) split rgb/ir
#         2) per-modality global descriptor -> small MLP -> per-channel scores
#         3) 对两个模态在同一通道上做 softmax（模态间竞争）
#         4) 用 weights 放缩并保留 residual（(1 + w * gamma)）
#         5) 拼接后 1x1 conv + BN，可选轻量空间 attention
#         6) residual add to input -> return (shape 保持不变)
#     """
#     def __init__(self, c1, r=16, dimension=1):
#         super().__init__()
#         # keep provided c1 but do NOT assume it's single-modality channels until forward
#         self.c1_provided = int(c1)
#         self.r = max(1, int(r))
#         self.dimension = dimension

#         # lazy-init flags; actual submodules constructed when forward sees x
#         self._initialized = False

#         # allow user to toggle spatial attention after construction if desired:
#         # e.g. model.module.RIFusion_instance.use_spatial = False
#         self.use_spatial = True

#     def _init_layers(self, half_c):
#         """Internal: build submodules once we know single-modality channel count"""
#         self.half_c = int(half_c)
#         self.in_c = self.half_c * 2

#         hidden = max(self.half_c // self.r, 1)

#         # small per-modality MLPs (operating on global pooled channel descriptors)
#         # using Linear layers (applied on descriptors of size (B, C))
#         self.mlp_rgb = nn.Sequential(
#             nn.Linear(self.half_c, hidden, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, self.half_c, bias=False)
#         )
#         self.mlp_ir = nn.Sequential(
#             nn.Linear(self.half_c, hidden, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, self.half_c, bias=False)
#         )

#         # global pool used to create descriptors
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         # channel mixing after modality weighting: light 1x1 conv + BN
#         self.fusion_conv = nn.Conv2d(self.in_c, self.in_c, kernel_size=1, stride=1, padding=0, bias=False)
#         self.fusion_bn = nn.BatchNorm2d(self.in_c)

#         # lightweight spatial attention (optional, default enabled)
#         mid = max(self.in_c // self.r, 4)
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(self.in_c, mid, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid, 1, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )

#         # learnable scale to control how strongly weighted modalities perturb the original
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self._initialized = True

#     def forward(self, x):
#         """
#         x: (B, C_in, H, W)
#         returns: (B, C_in, H, W) with same channel order as input
#         """
#         b, c_in, h, w = x.shape

#         # lazy init on first forward: infer half_c from provided c1 and actual x channels
#         if not self._initialized:
#             # cases to handle:
#             # 1) user passed single-modal c1 (C) and x has 2*C -> fine
#             # 2) user passed total c1 (2*C) and x has 2*C -> detect and use half = c1//2
#             # 3) other mismatch -> fallback to half = x.shape[1] // 2
#             if self.c1_provided == c_in:
#                 # user passed total channels: e.g., c1=128 and x has 128 -> half = 64
#                 half = c_in // 2
#             elif self.c1_provided * 2 == c_in:
#                 # user passed single-modality channels: c1=64 and x has 128
#                 half = self.c1_provided
#             else:
#                 # fallback heuristic: assume x channels are double and split in half
#                 half = c_in // 2
#             # build internal layers
#             self._init_layers(half)

#         # sanity check
#         if c_in != self.in_c:
#             # if still mismatched, try to adapt (rare) but error explicitly
#             raise RuntimeError(f"RIFusion initialized for {self.in_c} channels but got input with {c_in} channels.")

#         # split RGB and IR (assumes concatenation order is [rgb, ir])
#         rgb, ir = torch.split(x, self.half_c, dim=1)  # each (B, C, H, W)

#         # channel descriptors via global avg pool -> (B, C)
#         d_rgb = self.avg_pool(rgb).view(b, self.half_c)
#         d_ir  = self.avg_pool(ir).view(b, self.half_c)

#         # per-modality scores per channel
#         s_rgb = self.mlp_rgb(d_rgb)  # (B, C)
#         s_ir  = self.mlp_ir(d_ir)    # (B, C)

#         # modality competition: softmax over modality axis for each channel
#         scores = torch.stack([s_rgb, s_ir], dim=-1)  # (B, C, 2)
#         weights = F.softmax(scores, dim=-1)  # (B, C, 2)

#         wr = weights[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
#         wi = weights[..., 1].unsqueeze(-1).unsqueeze(-1)

#         # apply weights but keep residual path with learnable small scale gamma
#         rgb_att = rgb * (1.0 + wr * self.gamma)
#         ir_att  = ir  * (1.0 + wi * self.gamma)

#         # concatenate and channel-mix
#         fused = torch.cat([rgb_att, ir_att], dim=1)  # (B, 2C, H, W)
#         fused = self.fusion_conv(fused)
#         fused = self.fusion_bn(fused)

#         # optional lightweight spatial attention to suppress local noise
#         if getattr(self, "use_spatial", True):
#             sa = self.spatial_att(fused)  # (B,1,H,W)
#             fused = fused * sa + fused  # residual style

#         # final residual add (preserve original information + learned fusion)
#         out = x + fused

#         return out


# file: balanced_c2f.py
# from typing import Optional

# class C2f(nn.Module):
#     """
#     Balanced C2f: 与原 C2f 接口兼容的改进版
#     支持可选的类别感知输入 cls_present (B, num_classes) 或 cls_prob (B, num_classes)
#     如果在 __init__ 中传入 num_classes，会自动创建 class2channel 映射投影。
#     """
#     def __init__(self, c1, c2=None, n=1, shortcut=True, g=1, e=0.5, num_classes: Optional[int]=None, freq_factor: Optional[dict]=None):
#         """
#         参数：
#           c1, c2, n, shortcut, g, e 与原 C2f 保持兼容
#           num_classes: 若提供，则启用类别到通道的线性映射（训练时可传 batch 的类别向量）
#           freq_factor: 可选 dict {class_idx: weight}，用于初始化 class projector 的偏置（弱监督）
#         """
#         super().__init__()
#         if c2 is None:
#             c2 = c1
#         self.c1 = c1
#         self.c2 = c2
#         self.n = n
#         self.shortcut = shortcut
#         self.g = g
#         self.e = e
#         self.c = int(c2 * e)  # 中间通道

#         # 与原 C2f 逻辑一致的子模块（保证行为兼容）
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.m = nn.ModuleList([Conv(self.c, self.c, 3, 1, g=g) for _ in range(n)])
#         self.cv2 = Conv((n + 2) * self.c, c2, 1, 1)

#         # 通道注意力（SE-like），作用在 out 上
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c2, max(c2 // 16, 4), 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(max(c2 // 16, 4), c2, 1, bias=False),
#             nn.Sigmoid()
#         )

#         # 可选：类别 -> 通道 投影（用于把类别信息映射到通道偏好）
#         self.num_classes = num_classes
#         if num_classes is not None:
#             self.class_projector = nn.Linear(num_classes, c2, bias=True)
#             # 若提供 freq_factor (dict)，用它初始化 bias，使稀有类倾向于更大响应
#             if freq_factor is not None:
#                 device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device('cpu')
#                 # 构造初始类 prior 向量（长度 = num_classes）
#                 prior = torch.zeros(num_classes, device=device)
#                 for i in range(num_classes):
#                     prior[i] = float(freq_factor.get(i, 1.0))
#                 # normalize and convert to log-scale bias initialization
#                 prior = prior / (prior.mean() + 1e-9)
#                 init_bias = torch.log(prior + 1e-6)
#                 # set bias of projector to project average class prior to channel bias
#                 with torch.no_grad():
#                     self.class_projector.bias.copy_(init_bias.mean() * torch.ones(c2))
#         else:
#             self.class_projector = None

#         # 可学习缩放 gamma，初始 0（渐进式引入类别影响）
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x, cls_present: Optional[torch.Tensor]=None):
#         """
#         x: (B, C_in, H, W)
#         cls_present: 可选 (B, num_classes) 的 one-hot/频率/概率向量，表示该 batch/image 中的类别存在或概率。
#                      若提供，则用 class_projector 映射到 (B, C_out)，进一步影响通道注意力。
#         返回： (B, C_out, H, W)
#         兼容原 C2f 的用法（若不传 cls_present 则行为退化为加入 SE-style attention 的 C2f）
#         """
#         b, _, h, w = x.shape
#         y = list(self.cv1(x).chunk(2, 1))  # [x1, x2] each (B, c, H, W)
#         for m in self.m:
#             y.append(m(y[-1]))
#         out = self.cv2(torch.cat(y, 1))  # (B, C_out, H, W)

#         # 通道 attention
#         w = self.ca(out)  # (B, C_out, 1, 1)

#         # 如果提供了类别信息并且 projector 可用，则将类别投影到通道并作为额外增益
#         if cls_present is not None and self.class_projector is not None:
#             # cls_present shape (B, num_classes)
#             proj = self.class_projector(cls_present)  # (B, C_out)
#             proj = torch.sigmoid(proj).view(b, self.c2, 1, 1)  # (B, C,1,1)
#             # w * (1 + gamma * proj)  -> gamma 初始 0，训练中自适应放大
#             w = w * (1.0 + self.gamma * proj)

#         # 应用通道注意力（带残差风格）
#         out = out * w

#         return out


# class C2f_Faster(nn.Module):
#     """
#     更轻量 / 更快版本（减少中间 conv 重复、使用 depthwise-like reductions）
#     与 BalancedC2f 接口兼容（相同 __init__ 参数签名）
#     """
#     def __init__(self, c1, c2=None, n=1, shortcut=True, g=1, e=0.5, num_classes: Optional[int]=None, freq_factor: Optional[dict]=None):
#         super().__init__()
#         if c2 is None:
#             c2 = c1
#         self.c1 = c1
#         self.c2 = c2
#         self.n = n
#         self.shortcut = shortcut
#         self.e = e
#         self.c = int(c2 * e)

#         # Faster variant: first 1x1 reduce, then a single depthwise-like 3x3, then expand
#         self.reduce = Conv(c1, self.c * 2, 1, 1)
#         # depthwise-ish: use groups = self.c to reduce cost (approx depthwise)
#         self.dw = nn.Conv2d(self.c * 2, self.c * 2, kernel_size=3, stride=1, padding=1, groups=self.c * 2, bias=False)
#         self.dw_bn = nn.BatchNorm2d(self.c * 2)
#         self.act = nn.SiLU()
#         # expand back with 1x1
#         self.expand = Conv(self.c * 2, c2, 1, 1)

#         # attention & class projector as before
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c2, max(c2 // 16, 4), 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(max(c2 // 16, 4), c2, 1, bias=False),
#             nn.Sigmoid()
#         )
#         self.num_classes = num_classes
#         if num_classes is not None:
#             self.class_projector = nn.Linear(num_classes, c2, bias=True)
#             if freq_factor is not None:
#                 device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device('cpu')
#                 prior = torch.zeros(num_classes, device=device)
#                 for i in range(num_classes):
#                     prior[i] = float(freq_factor.get(i, 1.0))
#                 prior = prior / (prior.mean() + 1e-9)
#                 init_bias = torch.log(prior + 1e-6)
#                 with torch.no_grad():
#                     self.class_projector.bias.copy_(init_bias.mean() * torch.ones(c2))
#         else:
#             self.class_projector = None
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x, cls_present: Optional[torch.Tensor]=None):
#         """
#         x: (B, C_in, H, W)
#         cls_present: optional (B, num_classes)
#         """
#         b = x.shape[0]
#         x = self.reduce(x)  # (B, 2*c, H, W)
#         x = self.dw_bn(self.dw(x))
#         x = self.act(x)
#         out = self.expand(x)  # (B, c2, H, W)

#         w = self.ca(out)  # (B, C, 1, 1)
#         if cls_present is not None and self.class_projector is not None:
#             proj = self.class_projector(cls_present)
#             proj = torch.sigmoid(proj).view(b, self.c2, 1, 1)
#             w = w * (1.0 + self.gamma * proj)
#         out = out * w
#         return out





class CAFFBlock(nn.Module):
    def __init__(self, c1, c2=None, r=16, dimension=1):
        """
        改进的 Cross-Attention Fusion 模块
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数，如果为None则等于c1
            r (int): 通道缩减比
            dimension (int): 保持兼容性
        """
        super().__init__()
        self.r = r
        self.dimension = dimension
        
        # 延迟初始化标记
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = None
        self.spatial_conv = None
        self.cross_attn = None
        self.output_proj = None
        self.initialized = False
    
    def _initialize(self, channels):
        """动态初始化"""
        self.total_c = channels
        
        # 确保通道数合理分割
        self.c1 = channels // 2 if channels % 2 == 0 else (channels + 1) // 2
        self.c2 = channels - self.c1
        
        # 1. 通道注意力 - 使用单独的avg和max分支
        self.fc_avg = nn.Sequential(
            nn.Linear(self.total_c, self.total_c // self.r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.total_c // self.r, self.total_c, bias=False)
        )
        
        self.fc_max = nn.Sequential(
            nn.Linear(self.total_c, self.total_c // self.r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.total_c // self.r, self.total_c, bias=False)
        )
        
        # 2. 空间注意力
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 3. 交叉注意力融合
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=min(self.c1, self.c2),
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 4. 输出投影
        self.output_proj = nn.Conv2d(
            self.c1 + self.c2, 
            self.total_c, 
            kernel_size=1, 
            bias=False
        )
        
        # 5. 残差连接的权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        # 权重初始化
        self._init_weights()
        self.initialized = True
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        if not self.initialized:
            self._initialize(c)
        
        # 1. 通道注意力
        ch_avg = self.avg_pool(x).view(b, c)
        ch_max = F.adaptive_max_pool2d(x, 1).view(b, c)
        
        ch_attn_avg = self.fc_avg(ch_avg)
        ch_attn_max = self.fc_max(ch_max)
        ch_attn = torch.sigmoid(ch_attn_avg + ch_attn_max).view(b, c, 1, 1)
        
        x_ch = x * ch_attn
        
        # 2. 空间注意力
        sp_avg = x_ch.mean(dim=1, keepdim=True)
        sp_max = x_ch.max(dim=1, keepdim=True)[0]
        sp_concat = torch.cat([sp_avg, sp_max], dim=1)
        sp_attn = self.spatial_conv(sp_concat)
        
        x_sp = x_ch * sp_attn
        
        # 3. 特征分割
        feat_1 = x_sp[:, :self.c1]  # [B, C1, H, W]
        feat_2 = x_sp[:, self.c1:self.c1+self.c2]  # [B, C2, H, W]
        
        # 4. 交叉注意力融合（如果通道数匹配）
        if self.c1 == self.c2:
            # 将特征图reshape为序列进行attention
            feat_1_seq = feat_1.flatten(2).transpose(1, 2)  # [B, HW, C1]
            feat_2_seq = feat_2.flatten(2).transpose(1, 2)  # [B, HW, C2]
            
            # 交叉注意力
            feat_1_attn, _ = self.cross_attn(feat_1_seq, feat_2_seq, feat_2_seq)
            feat_2_attn, _ = self.cross_attn(feat_2_seq, feat_1_seq, feat_1_seq)
            
            # 重新reshape回特征图
            feat_1_attn = feat_1_attn.transpose(1, 2).view(b, self.c1, h, w)
            feat_2_attn = feat_2_attn.transpose(1, 2).view(b, self.c2, h, w)
            
            # 残差连接
            feat_1 = feat_1 + feat_1_attn
            feat_2 = feat_2 + feat_2_attn
        
        # 5. 特征融合
        fused = torch.cat([feat_1, feat_2], dim=1)
        
        # 6. 输出投影
        out = self.output_proj(fused)
        
        # 7. 残差连接
        out = self.alpha * out + (1 - self.alpha) * x
        
        return out
# class CAFFBlock(nn.Module):
#     """
#     改进的跨模态自适应特征融合块 (Improved Cross-modal Adaptive Feature Fusion)
    
#     主要改进：
#     1. 多尺度空间注意力 - 更好的空间特征感知
#     2. 改进的通道注意力 - 结合平均池化和最大池化
#     3. 模态权重学习 - 自适应调节RGB和IR的贡献
#     4. 残差连接 - 防止梯度消失
#     5. 特征对齐 - 确保两个模态特征分布一致
#     """
    
#     def __init__(self, channels, reduction=16):
#         """
#         Args:
#             channels: 输入特征的通道数 (对应原始CAFFBlock的第一个参数)
#             reduction: 通道注意力的降维比例
#         """
#         super().__init__()
#         self.channels = channels
        
#         # 1. 特征对齐层 - 确保RGB和IR特征分布一致
#         self.align_rgb = nn.Sequential(
#             nn.Conv2d(channels, channels, 1, bias=False),
#             nn.BatchNorm2d(channels)
#         )
#         self.align_ir = nn.Sequential(
#             nn.Conv2d(channels, channels, 1, bias=False),
#             nn.BatchNorm2d(channels)
#         )
        
#         # 2. 改进的通道注意力 - 结合GAP和GMP
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.gmp = nn.AdaptiveMaxPool2d(1)
        
#         # 通道注意力网络
#         self.channel_attention = nn.Sequential(
#             nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False),
#             nn.Sigmoid()
#         )
        
#         # 3. 多尺度空间注意力
#         self.spatial_conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)  # 3x3
#         self.spatial_conv2 = nn.Conv2d(2, 1, 5, padding=2, bias=False)  # 5x5
#         self.spatial_conv3 = nn.Conv2d(2, 1, 7, padding=3, bias=False)  # 7x7
#         self.spatial_fusion = nn.Conv2d(3, 1, 1, bias=False)
#         self.spatial_sigmoid = nn.Sigmoid()
        
#         # 4. 模态权重学习 - 自适应调节RGB和IR贡献
#         self.modal_weight_fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels * 2, channels // 4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 4, 2, 1),  # 输出2个权重：RGB和IR
#             nn.Softmax(dim=1)
#         )
        
#         # 5. 特征融合层
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU(inplace=True)
#         )
        
#         # 6. 残差连接的投影层
#         self.residual_proj = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 1, bias=False),
#             nn.BatchNorm2d(channels)
#         )
        
#         # 7. 最终激活
#         self.final_act = nn.SiLU(inplace=True)
        
#     def forward(self, rgb_feat, ir_feat):
#         """
#         Args:
#             rgb_feat: RGB特征 [B, C, H, W]
#             ir_feat: IR特征 [B, C, H, W]
#         Returns:
#             融合后的特征 [B, C, H, W]
#         """
#         # 1. 特征对齐
#         rgb_aligned = self.align_rgb(rgb_feat)
#         ir_aligned = self.align_ir(ir_feat)
        
#         # 2. 特征连接
#         combined = torch.cat([rgb_aligned, ir_aligned], dim=1)
        
#         # 3. 通道注意力计算
#         gap_feat = self.gap(combined)  # [B, 2C, 1, 1]
#         gmp_feat = self.gmp(combined)  # [B, 2C, 1, 1]
#         channel_input = torch.cat([gap_feat, gmp_feat], dim=1)  # [B, 4C, 1, 1]
#         channel_weight = self.channel_attention(channel_input)  # [B, C, 1, 1]
        
#         # 4. 多尺度空间注意力计算
#         # 计算空间统计信息
#         spatial_avg = torch.mean(combined, dim=1, keepdim=True)  # [B, 1, H, W]
#         spatial_max = torch.max(combined, dim=1, keepdim=True)[0]  # [B, 1, H, W]
#         spatial_input = torch.cat([spatial_avg, spatial_max], dim=1)  # [B, 2, H, W]
        
#         # 多尺度卷积
#         spatial_feat1 = self.spatial_conv1(spatial_input)  # 3x3
#         spatial_feat2 = self.spatial_conv2(spatial_input)  # 5x5  
#         spatial_feat3 = self.spatial_conv3(spatial_input)  # 7x7
        
#         # 融合多尺度空间特征
#         spatial_combined = torch.cat([spatial_feat1, spatial_feat2, spatial_feat3], dim=1)
#         spatial_weight = self.spatial_sigmoid(self.spatial_fusion(spatial_combined))
        
#         # 5. 模态权重学习
#         modal_weights = self.modal_weight_fc(combined)  # [B, 2, 1, 1]
#         rgb_weight = modal_weights[:, 0:1, :, :]  # [B, 1, 1, 1]
#         ir_weight = modal_weights[:, 1:2, :, :]   # [B, 1, 1, 1]
        
#         # 6. 应用注意力权重
#         weighted_rgb = rgb_aligned * channel_weight * spatial_weight * rgb_weight
#         weighted_ir = ir_aligned * channel_weight * spatial_weight * ir_weight
        
#         # 7. 特征融合
#         weighted_combined = torch.cat([weighted_rgb, weighted_ir], dim=1)
#         fused_feat = self.fusion_conv(weighted_combined)
        
#         # 8. 残差连接
#         residual = self.residual_proj(combined)
#         output = fused_feat + residual
        
#         return self.final_act(output)
class GateBlock(nn.Module):
    def __init__(self, c1, temp=1.0, eps=1e-10):
        """
        Simplified and stable dynamic channel gating module
        
        Args:
            c1: Input channels (as specified in YAML config) - will be ignored, auto-detected
            temp: Gumbel-Softmax temperature (lower = harder selection)
            eps: Small constant for numerical stability
        """
        super().__init__()
        # Store parameters
        self.temp = temp
        self.eps = eps
        
        # 不预先创建任何需要通道数的层，全部在forward中动态创建
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_layers = nn.ModuleDict()
        self.alpha = nn.Parameter(torch.tensor(0.8)) 
        self.current_channels = {}
        
        print(f"GateBlock initialized (dynamic channels)")
    
    def _get_or_create_linear(self, channels, device):
        """动态获取或创建线性层"""
        key = str(channels)
        if key not in self.linear_layers:
            linear = nn.Linear(channels, channels).to(device)
            # 初始化权重
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linear_layers[key] = linear
            print(f"Created linear layer for {channels} channels")
        return self.linear_layers[key]
    
    def forward(self, x):
        """
        Simplified forward pass with automatic channel detection
        
        Args:
            x: Input tensor or list of tensors
            
        Returns:
            Gated features with same shape as input
        """
        # 处理列表输入（来自ADD操作）
        if isinstance(x, list):
            # 如果是ADD操作的结果，通常第一个元素是结果
            if len(x) == 1:
                return self._process_tensor(x[0])
            
            # 简单的张量融合 - 确保维度匹配
            target_tensor = x[0]
            for tensor in x[1:]:
                if tensor.shape == target_tensor.shape:
                    target_tensor = target_tensor + tensor
                else:
                    # 如果维度不匹配，使用插值对齐
                    tensor_resized = F.interpolate(
                        tensor, 
                        size=target_tensor.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                    target_tensor = target_tensor + tensor_resized
            
            return self._process_tensor(target_tensor)
        
        return self._process_tensor(x)
    
    def _process_tensor(self, x):
        """简化的张量处理"""
        B, C, H, W = x.shape
        device = x.device
        
        # 1. 池化
        pooled = self.avg_pool(x)
        flattened = pooled.view(B, C)
        
        # 2. 动态线性层
        linear = self._get_or_create_linear(C, device)
        
        # 3. 门控权重
        gate_weights = torch.sigmoid(linear(flattened))
        
        # 4. 训练噪声
        if self.training:
            noise = torch.randn_like(gate_weights) * 0.01
            gate_weights = torch.clamp(gate_weights + noise, 0, 1)
        
        # 5. 应用门控
        gate_weights = gate_weights.view(B, C, 1, 1)
        gated_output = x * gate_weights
        
        # 6. 可学习残差连接
        output = torch.sigmoid(self.alpha) * gated_output + (1 - torch.sigmoid(self.alpha)) * x
        return output

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional

# # 1. 改进的注意力机制 - ECA注意力（轻量级且有效）
# class ECABlock(nn.Module):
#     """Efficient Channel Attention Block"""
#     def __init__(self, c1, gamma=2, b=1):
#         super().__init__()
#         k = int(abs((math.log(c1, 2) + b) / gamma))
#         k = k if k % 2 else k + 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)

# # 2. 改进的RGB-IR融合模块 - 自适应加权融合
# class AdaptiveFusionBlock(nn.Module):
#     """Adaptive weighted fusion for RGB and IR features"""
#     def __init__(self, c1, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(c1, c1 // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c1 // reduction, c1, bias=False),
#             nn.Sigmoid()
#         )
#         self.conv_fusion = nn.Conv2d(c1 * 2, c1, 1, bias=False)
#         self.bn = nn.BatchNorm2d(c1)
#         self.act = nn.SiLU()
        
#     def forward(self, rgb_feat, ir_feat):
#         # 计算自适应权重
#         combined = torch.cat([rgb_feat, ir_feat], dim=1)
#         b, c, _, _ = combined.size()
#         y = self.avg_pool(combined).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
        
#         # 分离RGB和IR权重
#         rgb_weight, ir_weight = y[:, :c//2], y[:, c//2:]
        
#         # 加权融合
#         weighted_rgb = rgb_feat * rgb_weight
#         weighted_ir = ir_feat * ir_weight
        
#         # 特征融合
#         fused = torch.cat([weighted_rgb, weighted_ir], dim=1)
#         fused = self.conv_fusion(fused)
#         fused = self.bn(fused)
#         fused = self.act(fused)
        
#         return fused

# # 3. 改进的C2f模块 - 增加注意力
# class C2f_ECA(nn.Module):
#     """C2f module with ECA attention"""
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#         self.eca = ECABlock(c2)
        
#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         out = self.cv2(torch.cat(y, 1))
#         return self.eca(out)

# # 4. 多尺度特征增强模块
# class ASFF(nn.Module):
#     """Adaptive Spatial Feature Fusion"""
#     def __init__(self, level, rfb=False, vis=False):
#         super(ASFF, self).__init__()
#         self.level = level
#         self.dim = [512, 256, 128]
#         self.inter_dim = self.dim[self.level]
        
#         if level == 0:
#             self.stride_level_1 = Conv(256, self.inter_dim, 3, 2)
#             self.stride_level_2 = Conv(128, self.inter_dim, 3, 2)
#             self.expand = Conv(self.inter_dim, 512, 3, 1)
#         elif level == 1:
#             self.compress_level_0 = Conv(512, self.inter_dim, 1, 1)
#             self.stride_level_2 = Conv(128, self.inter_dim, 3, 2)
#             self.expand = Conv(self.inter_dim, 256, 3, 1)
#         elif level == 2:
#             self.compress_level_0 = Conv(512, self.inter_dim, 1, 1)
#             self.compress_level_1 = Conv(256, self.inter_dim, 1, 1)
#             self.expand = Conv(self.inter_dim, 128, 3, 1)
            
#         self.weight_level_0 = Conv(self.inter_dim, 1, 1, 1)
#         self.weight_level_1 = Conv(self.inter_dim, 1, 1, 1)
#         self.weight_level_2 = Conv(self.inter_dim, 1, 1, 1)
#         self.weight_levels = Conv(self.inter_dim * 3, 3, 1, 1)
        
#     def forward(self, x):
#         level_0_resized, level_1_resized, level_2_resized = x
        
#         if self.level == 0:
#             level_1_resized = self.stride_level_1(level_1_resized)
#             level_2_resized = self.stride_level_2(level_2_resized)
#         elif self.level == 1:
#             level_0_resized = F.interpolate(self.compress_level_0(level_0_resized), 
#                                           scale_factor=2, mode='nearest')
#             level_2_resized = self.stride_level_2(level_2_resized)
#         elif self.level == 2:
#             level_0_resized = F.interpolate(self.compress_level_0(level_0_resized), 
#                                           scale_factor=4, mode='nearest')
#             level_1_resized = F.interpolate(self.compress_level_1(level_1_resized), 
#                                           scale_factor=2, mode='nearest')
            
#         level_0_weight = self.weight_level_0(level_0_resized)
#         level_1_weight = self.weight_level_1(level_1_resized)
#         level_2_weight = self.weight_level_2(level_2_resized)
        
#         levels_weight = torch.cat((level_0_weight, level_1_weight, level_2_weight), 1)
#         levels_weight = self.weight_levels(levels_weight)
#         levels_weight = F.softmax(levels_weight, dim=1)
        
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                            level_1_resized * levels_weight[:, 1:2, :, :] + \
#                            level_2_resized * levels_weight[:, 2:, :, :]
        
#         out = self.expand(fused_out_reduced)
#         return out

# 5. 改进的SPPF模块 - 增加更多尺度
# class SPPF(nn.Module):
#     """Enhanced SPPF with more pooling scales"""
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_ = c1 // 2
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 5, c2, 1, 1)  # 增加到5个分支
#         self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)    # 新增
#         self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)   # 新增
        
#     def forward(self, x):
#         x = self.cv1(x)
#         y1 = self.m1(x)
#         y2 = self.m1(y1)
#         y3 = self.m2(x)      # 新的池化分支
#         y4 = self.m3(x)      # 新的池化分支
#         return self.cv2(torch.cat((x, y1, y2, y3, y4), 1))


class CAFFBlock(nn.Module):
    """Adaptive weighted fusion for RGB and IR features"""
    def __init__(self, c1, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1 * 2, c1 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, 2, bias=False),
            nn.Sigmoid()
        )
        self.conv_fusion = Conv(c1 * 2, c1, 1, 1)
        
    def forward(self, x):
        # x是包含两个特征的列表 [rgb_feat, ir_feat]
        rgb_feat, ir_feat = x if isinstance(x, (list, tuple)) else (x, x)
        
        # 计算自适应权重
        combined = torch.cat([rgb_feat, ir_feat], dim=1)
        b, c, h, w = combined.size()
        
        # 全局平均池化获取权重
        y = self.avg_pool(combined).view(b, c)
        weights = self.fc(y).view(b, 2, 1, 1)
        
        # 加权融合
        weighted_rgb = rgb_feat * weights[:, 0:1]
        weighted_ir = ir_feat * weights[:, 1:2]
        
        # 特征融合
        fused = torch.cat([weighted_rgb, weighted_ir], dim=1)
        fused = self.conv_fusion(fused)
        
        return fused





# class CAFFBlock(nn.Module):
#     """Channel Attention Feature Fusion Block"""
#     def __init__(self, c1, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(c1, c1 // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(c1 // reduction, c1, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return x * self.sigmoid(out)

# #成功51     R=0.775
# import torch
# import torch.nn as nn

# import torch
# import torch.nn as nn

# class C2f(nn.Module):
#     """
#     Enhanced C2f module integrating:
#       - Fine-grained Multi-scale Dynamic Selection (FMDS)
#       - Adaptive Gated Multi-branch Fusion (AGMF)
#       - Efficient Local Attention (ELA)
#       - Sliding-window Cross Convolution (CrossConv)
#       - Instance-Specific Bottleneck with matching channels for modulation
#     """
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels

#         # 1. Multi-branch feature extraction
#         self.branch_spatial = nn.Sequential(
#             nn.Conv2d(c1, c_, 1, 1, bias=False),
#             nn.BatchNorm2d(c_),
#             nn.ReLU(inplace=True)
#         )
#         self.branch_channel = nn.Sequential(
#             nn.Conv2d(c1, c_, 1, 1, groups=c_, bias=False),
#             nn.BatchNorm2d(c_),
#             nn.ReLU(inplace=True)
#         )
#         self.branch_cross = nn.Sequential(
#             nn.Conv2d(c1, c_, 3, 1, 1, groups=g, bias=False),
#             nn.BatchNorm2d(c_),
#             nn.ReLU(inplace=True)
#         )

#         # 2. Adaptive gating to fuse branches
#         self.gate = nn.Sequential(
#             nn.Conv2d(c_ * 3, c_ * 3, 1, 1, bias=False),
#             nn.Sigmoid()
#         )

#         # 3. Efficient Local Attention
#         self.local_attn = nn.Sequential(
#             nn.Conv2d(c_ * 3, c_ * 3, 1, 1, bias=False),
#             nn.BatchNorm2d(c_ * 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c_ * 3, c_ * 3, 3, 1, padding=1, groups=c_ * 3, bias=False),
#             nn.BatchNorm2d(c_ * 3),
#             nn.Sigmoid()
#         )

#         # 4. Instance-Specific Bottleneck (matching hidden channels)
#         self.isb = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c_ * 3, c_ * 3, 1, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c_ * 3, c_ * 3, 1, 1, bias=False),
#             nn.Sigmoid()
#         )

#         # 5. Final projection
#         self.conv_out = nn.Conv2d(c_ * 3, c2, 1, 1, bias=False)

#         # shortcut
#         self.shortcut = shortcut and c1 == c2

#     def forward(self, x):
#         # 1. multi-branch
#         f1 = self.branch_spatial(x)
#         f2 = self.branch_channel(x)
#         f3 = self.branch_cross(x)

#         # 2. concatenate and gate
#         fused = torch.cat([f1, f2, f3], dim=1)
#         gated = fused * self.gate(fused)

#         # 3. local attention
#         attended = gated * self.local_attn(gated)

#         # 4. instance-specific modulation
#         modulated = attended * self.isb(attended)

#         # 5. output projection
#         out = self.conv_out(modulated)

#         # 6. residual
#         return out + x if self.shortcut else out





class C3k2():
    a=1




import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Create spatial attention mask
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        
        # Ensure spatial dimensions match
        if y.shape[2:] != x.shape[2:]:
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return self.sigmoid(y)


#@register_block  # Add this line to register the
# 
# 
# 
#  module
class CBAMk(nn.Module):
    """
    Convolutional Block Attention Module with kernel size specification
    """
    def __init__(self, c1, c2=None):  # c2 is unused but kept for compatibility with other modules
        super().__init__()
        self.channel_attention = ChannelAttention(c1, ratio=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
    




# #放到 ultralytics/nn/modules/attention.py 中
# class EMA(nn.Module):
#     """
#     Efficient Channel Attention Module (ECA)

#     Args:
#         channels (int): Number of input channels.
#         k_size (int, optional): Kernel size for adaptive 1D convolution. Must be odd. Default: 3.
#     """
#     def __init__(self, channels: int, k_size: int = 3):
#         super(EMA, self).__init__()
#         assert k_size % 2 == 1, "k_size must be odd"
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # 1D convolution with padding
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, C, H, W]
#         # Squeeze: global average pooling -> [B, C, 1, 1]
#         y = self.avg_pool(x)
#         # Reshape for conv1d: [B, 1, C]
#         y = y.view(x.size(0), 1, x.size(1))
#         # Apply conv1d
#         y = self.conv(y)
#         # Activation
#         y = self.sigmoid(y)
#         # Reshape: [B, C, 1, 1]
#         y = y.view(x.size(0), x.size(1), 1, 1)
#         # Scale
#         return x * y

class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module (Channel + Spatial)
    Lightweight implementation suitable for YOLO modules.
    Args:
        channels: int, input channels
        reduction: int, channel reduction ratio for MLP
        kernel_size: int, conv kernel for spatial attention (usually 7)
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channels = channels
        mid = max(1, channels // reduction)
        # channel attention MLP using Conv1x1
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, stride=1, bias=True),
        )
        # spatial attention conv
        padding = (kernel_size - 1) // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)  # (B, C, 1, 1)
        ca = self.mlp(avg_pool) + self.mlp(max_pool)
        ca = torch.sigmoid(ca)  # (B, C, 1, 1)
        x_ca = x * ca

        # Spatial attention
        avg_c = torch.mean(x_ca, dim=1, keepdim=True)  # (B,1,H,W)
        max_c, _ = torch.max(x_ca, dim=1, keepdim=True)  # (B,1,H,W)
        sa_in = torch.cat([avg_c, max_c], dim=1)  # (B,2,H,W)
        sa = torch.sigmoid(self.spatial(sa_in))  # (B,1,H,W)

        out = x_ca * sa  # broadcast multiply
        return out


try:
    from spikingjelly.clock_driven.neuron import LIFNode
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False
    print("Warning: spikingjelly not installed, using fallback implementation")

class EnhancedSpikeAttention(nn.Module):
    def __init__(self, channels, num_heads=4, T=6):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.T = T
        
        # 多头QKV投影
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_bn = nn.BatchNorm2d(channels * 3)
        
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)
        self.tau_q = nn.Parameter(torch.ones(num_heads) * 2.0)
        self.tau_k = nn.Parameter(torch.ones(num_heads) * 2.0)
        self.tau_v = nn.Parameter(torch.ones(num_heads) * 2.0)
        self.threshold = nn.Parameter(torch.ones(1) * 0.5)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def create_lif_node(self, tau_value):
        if HAS_SPIKINGJELLY:
            return LIFNode(tau=tau_value.item())
        else:
            # Fallback: 简单的阈值激活
            class SimpleLIF:
                def __init__(self, tau):
                    self.tau = tau
                    self.v = 0
                def __call__(self, x):
                    self.v = self.v * (1 - 1/self.tau) + x
                    spike = (self.v > 0.5).float()
                    self.v = self.v * (1 - spike)
                    return spike
            return SimpleLIF(tau_value.item())
    
    def forward(self, x):
        B, C, H, W = x.shape
        dtype = x.dtype
        N = H * W
        
        # QKV投影
        qkv = self.qkv_bn(self.qkv(x))  # (B, 3C, H, W)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, N)  # (B, 3, heads, head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, N, head_dim)
        q0, k0, v0 = qkv[0], qkv[1], qkv[2]
        
        # 多头并行处理
        outputs = []
        for head in range(self.num_heads):
            # 为每个头创建独立的LIF节点
            lif_q = self.create_lif_node(self.tau_q[head])
            lif_k = self.create_lif_node(self.tau_k[head])
            lif_v = self.create_lif_node(self.tau_v[head])
            
            Q_acc = torch.zeros_like(q0[:, head])  # (B, N, head_dim)
            K_acc = torch.zeros_like(k0[:, head])
            V_acc = torch.zeros_like(v0[:, head])
            
            # 时间步累积
            for t in range(self.T):
                Q_spk = lif_q(q0[:, head])
                K_spk = lif_k(k0[:, head])
                V_spk = lif_v(v0[:, head])
                
                Q_acc += Q_spk
                K_acc += K_spk
                V_acc += V_spk
            
            #平均得到速率编码
            Q = Q_acc / self.T
            K = K_acc / self.T
            V = V_acc / self.T
            
            # 标准注意力机制
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            head_out = torch.matmul(attn_weights, V)  # (B, N, head_dim)
            
            outputs.append(head_out)
        
        # 多头拼接
        multi_head_out = torch.cat(outputs, dim=-1)  # (B, N, C)
        multi_head_out = multi_head_out.permute(0, 2, 1).view(B, C, H, W)
        
        # 输出投影
        out = self.out_bn(self.out_conv(multi_head_out))
        
        #残差连接
        return x + out * self.residual_scale.to(dtype)


class ESSA(nn.Module):
    def __init__(self, channels, num_heads=4, T=6):
        super().__init__()
        
        # 增强的脉冲注意力
        self.spike_attn = EnhancedSpikeAttention(channels, num_heads, T)
        
        # FFN增强表达能力
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # FFN残差缩放
        self.ffn_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        dtype = x.dtype
        
        # 1. 脉冲注意力分支
        attn_out = self.spike_attn(x)
        
        # 2. FFN分支
        ffn_out = self.ffn(attn_out)
        attn_out = attn_out + ffn_out * self.ffn_scale.to(dtype)
        
        # 3. 通道门控
        gate = self.channel_gate(attn_out)
        out = attn_out * gate
        
        return out.to(dtype)
