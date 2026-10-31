import math
import torch
from efod import build_model
from efod.evaluation.thesis_energy import estimate_thesis_energy


def test_thesis_energy_is_positive_for_ann_and_snn():
    for pipe in ('event_snn','strict_ann','native_ann'):
        m=build_model('vgg16',pipe,image_height=120,image_width=160).eval()
        x=torch.zeros(1,2 if pipe=='event_snn' else 3,120,160)
        if pipe=='event_snn': x[:,:,20:25,20:25]=1
        r=estimate_thesis_energy(m,x)
        assert math.isfinite(r['thesis_total_uJ']) and r['thesis_total_uJ']>0
        assert r['memory_energy_pj']>0
