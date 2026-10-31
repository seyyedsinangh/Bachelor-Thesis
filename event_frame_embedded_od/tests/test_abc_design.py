import torch
from torch import nn
from efod import build_model
from efod.models.blocks import PreActSpikingBlock, PreActANNBlock, NativeANNBlock


def test_abc_block_orders_are_explicit():
    a=build_model('vgg16','event_snn',image_height=120,image_width=160)
    b=build_model('vgg16','strict_ann',image_height=120,image_width=160)
    c=build_model('vgg16','native_ann',image_height=120,image_width=160)
    assert isinstance(a.backbone.core.blocks[0],PreActSpikingBlock)
    assert isinstance(b.backbone.core.blocks[0],PreActANNBlock)
    assert isinstance(c.backbone.core.blocks[0],NativeANNBlock)
    assert a.backbone.core.blocks[0].bn.num_features==2
    assert b.backbone.core.blocks[0].bn.num_features==3
    assert c.backbone.core.blocks[0].bn.num_features==16
    assert a.backbone.core.blocks[0].conv.in_channels==2
    assert b.backbone.core.blocks[0].conv.in_channels==3
    assert c.backbone.core.blocks[0].conv.in_channels==3


def test_rescat_published_core_progression():
    m=build_model('rescat16','event_snn',image_height=120,image_width=160)
    core=m.backbone.core
    assert core.patchify.conv.out_channels==16
    assert core.pre.conv.out_channels==32
    assert core.stage1.out_channels==48
    assert core.stage2.out_channels==96
    assert core.stage3.out_channels==96
    assert core.feature_channels==[48,96,96]
