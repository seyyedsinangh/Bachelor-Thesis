import torch
from efod import build_model

def test_three_training_forwards():
    for pipe in ('event_snn','strict_ann','native_ann'):
        m=build_model('vgg16',pipe,image_height=240,image_width=320).train(); x=torch.rand(2 if pipe=='event_snn' else 3,240,320)
        if pipe=='event_snn': x=(x>.99).float()
        tgt={'boxes':torch.tensor([[20.,20.,60.,70.]]),'labels':torch.tensor([3])}; losses=m([x],[tgt])
        assert set(losses)=={'bbox_regression','classification'}; assert all(torch.isfinite(v) for v in losses.values())
