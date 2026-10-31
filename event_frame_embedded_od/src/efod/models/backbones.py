from __future__ import annotations

from collections import OrderedDict
from torch import nn

from .blocks import make_block, ResCatStage


class STVGGBackbone(nn.Module):
    """Cordone ST-VGG core: 8 conv blocks, patchify stem, three detector taps.

    Event-SNN uses BN->Conv->PLIF. The two ANN pipelines preserve the same convolution
    widths, kernels, strides and taps while changing activation/BN ordering only.
    """
    def __init__(self, C: int, pipeline: str, input_channels: int, init_tau=2.0, threshold=1.0):
        super().__init__()
        channels = [C, 2*C, 2*C, 2*C, 2*C, 4*C, 4*C, 4*C]
        strides  = [4, 1, 1, 2, 1, 2, 1, 2]
        kernels  = [4, 3, 3, 3, 3, 3, 3, 3]
        paddings = [0, 1, 1, 1, 1, 1, 1, 1]
        blocks=[]; cin=input_channels
        for cout,k,s,p in zip(channels,kernels,strides,paddings):
            blocks.append(make_block(pipeline,cin,cout,k,s,p,init_tau=init_tau,threshold=threshold))
            cin=cout
        self.blocks=nn.ModuleList(blocks)
        # At 304x240 these taps are 38x30, 19x15 and 10x8 as in Fig. 5.8.
        self.tap_indices=(4,6,7)
        self.feature_channels=[2*C,4*C,4*C]

    def forward(self,x):
        out=OrderedDict(); j=0
        for i,b in enumerate(self.blocks):
            x=b(x)
            if i in self.tap_indices:
                out[str(j)]=x; j+=1
        return out


class ResCatBackbone(nn.Module):
    """Cordone ResCat-SNN core reconstructed from Figs. 5.6/5.7 and Table 5.2.

    Channel progression is C -> 2C -> 3C -> 6C -> 6C.  Each residual stage uses
    channel concatenation.  The hidden widths 2C, 3C, 6C are crucial: they reproduce
    the thesis Table-5.2 parameter counts (216k/862k/3.442M) at C=16/32/64.
    """
    def __init__(self,C:int,pipeline:str,input_channels:int,init_tau=2.0,threshold=1.0):
        super().__init__(); kw=dict(init_tau=init_tau,threshold=threshold)
        self.patchify=make_block(pipeline,input_channels,C,4,4,0,**kw)
        self.pre=make_block(pipeline,C,2*C,3,1,1,**kw)
        self.stage1=ResCatStage(pipeline,2*C,2*C,2*C,C,**kw)       # 2C + C = 3C
        self.stage2=ResCatStage(pipeline,3*C,3*C,4*C,2*C,**kw)     # 4C + 2C = 6C
        self.stage3=ResCatStage(pipeline,6*C,6*C,4*C,2*C,**kw)     # 4C + 2C = 6C
        self.feature_channels=[3*C,6*C,6*C]

    def forward(self,x):
        x=self.pre(self.patchify(x))
        f1=self.stage1(x); f2=self.stage2(f1); f3=self.stage3(f2)
        return OrderedDict([('0',f1),('1',f2),('2',f3)])


class SSDFeatureBackbone(nn.Module):
    """Core backbone + the three reduced 3x3/stride-2 SSD extra blocks from Ch. 5."""
    def __init__(self,core:nn.Module,pipeline:str,extra_channels=(256,128,128),init_tau=2.0,threshold=1.0):
        super().__init__(); self.core=core
        base=list(core.feature_channels); extras=[]; cin=base[-1]
        for cout in extra_channels:
            extras.append(make_block(pipeline,cin,cout,3,2,1,init_tau=init_tau,threshold=threshold)); cin=cout
        self.extras=nn.ModuleList(extras)
        self.out_channels=base+list(extra_channels)

    def forward(self,x):
        feats=self.core(x); out=OrderedDict(feats); y=list(feats.values())[-1]; idx=len(out)
        for b in self.extras:
            y=b(y); out[str(idx)]=y; idx+=1
        return out
