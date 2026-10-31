#!/usr/bin/env python
from __future__ import annotations
import json,torch
from efod import build_model,MODEL_SPECS,PIPELINES,PIPELINE_DISPLAY
from efod.models import count_spiking_neurons
EXPECTED={'vgg16':125484,'vgg32':499788,'vgg64':1994892,'rescat16':216399,'rescat32':862351,'rescat64':3442959}
rows=[]
for v in MODEL_SPECS:
    shapes=None
    for p in PIPELINES:
        m=build_model(v,p).eval(); x=torch.zeros(1,2 if p=='event_snn' else 3,480,640)
        with torch.no_grad(): f,h=m.forward_raw(x)
        fs=[list(z.shape) for z in f.values()]; core=sum(q.numel() for q in m.backbone.core.parameters())
        if p=='event_snn': assert core==EXPECTED[v]
        assert len(fs)==6 and h['cls_logits'].shape[-1]==8 and h['bbox_regression'].shape[-1]==4
        assert (count_spiking_neurons(m)>0)==(p=='event_snn')
        if shapes is None: shapes=[z[-2:] for z in fs]
        else: assert shapes==[z[-2:] for z in fs]
        rows.append({'variant':v,'pipeline':p,'pipeline_display':PIPELINE_DISPLAY[p],'core_params':core,'total_params':m.count_parameters(),'plif_modules':count_spiking_neurons(m),'feature_shapes':fs})
print(json.dumps(rows,indent=2)); print('\nPASS: all 18 models construct; A/B/C feature interfaces match; Event-SNN core counts match thesis reconstruction.')
