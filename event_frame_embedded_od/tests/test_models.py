import torch
from efod import build_model,MODEL_SPECS
from efod.models import count_spiking_neurons

# Exact counts from this clean reconstruction with 2-channel event input. Thesis Table 5.2 is rounded.
CORE_EVENT_COUNTS={'vgg16':125484,'vgg32':499788,'vgg64':1994892,'rescat16':216399,'rescat32':862351,'rescat64':3442959}

def test_thesis_backbone_parameter_regression():
    for v,expected in CORE_EVENT_COUNTS.items():
        m=build_model(v,'event_snn',image_height=120,image_width=160)
        got=sum(p.numel() for p in m.backbone.core.parameters())
        assert got==expected,(v,got,expected)
        rounded=MODEL_SPECS[v]['thesis_core_params']
        assert abs(got-rounded)/max(1,rounded)<0.005

def test_all_18_topology_parity_small():
    for v in MODEL_SPECS:
        shapes={}
        for pipe in ('event_snn','strict_ann','native_ann'):
            m=build_model(v,pipe,image_height=120,image_width=160).eval(); x=torch.zeros(1,2 if pipe=='event_snn' else 3,120,160)
            with torch.no_grad(): feats,head=m.forward_raw(x)
            shapes[pipe]=[tuple(t.shape[-2:]) for t in feats.values()]
            assert len(feats)==6; assert head['cls_logits'].shape[-1]==8; assert head['bbox_regression'].shape[-1]==4
            assert (count_spiking_neurons(m)>0)==(pipe=='event_snn')
        assert shapes['event_snn']==shapes['strict_ann']==shapes['native_ann']
