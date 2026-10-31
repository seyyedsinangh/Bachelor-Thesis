#!/usr/bin/env python
from __future__ import annotations
import argparse,json,torch
from efod import build_model,MODEL_SPECS,PIPELINES,PIPELINE_DISPLAY
from efod.models import count_spiking_neurons

def core_params(m): return sum(p.numel() for p in m.backbone.core.parameters())
def main():
    p=argparse.ArgumentParser(); p.add_argument('--variant',default='vgg32',choices=MODEL_SPECS); p.add_argument('--pipeline',default='event_snn',choices=PIPELINES); p.add_argument('--out',default=None); a=p.parse_args()
    m=build_model(a.variant,a.pipeline).eval(); x=torch.zeros(1,2 if a.pipeline=='event_snn' else 3,480,640)
    with torch.no_grad(): f,h=m.forward_raw(x)
    summary={'variant':a.variant,'display':MODEL_SPECS[a.variant]['display'],'pipeline':a.pipeline,'pipeline_display':PIPELINE_DISPLAY[a.pipeline],'params':m.count_parameters(),'core_params':core_params(m),'thesis_core_params_rounded':MODEL_SPECS[a.variant]['thesis_core_params'],'plif_modules':count_spiking_neurons(m),'feature_shapes':[list(t.shape) for t in f.values()],'cls_logits_shape':list(h['cls_logits'].shape),'bbox_regression_shape':list(h['bbox_regression'].shape)}
    text=json.dumps(summary,indent=2)+'\n\n'+str(m); print(text)
    if a.out: open(a.out,'w').write(text)
if __name__=='__main__': main()
