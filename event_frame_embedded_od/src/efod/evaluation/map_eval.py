from __future__ import annotations

import torch


def evaluate_map(model, loader, device='cuda', reset_each_sequence=True):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception as e:
        raise RuntimeError('Install torchmetrics[detection] / pycocotools for mAP evaluation') from e

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
    model.eval()
    if hasattr(model, 'reset_state'):
        model.reset_state()
    last_seq = None
    with torch.no_grad():
        for images, targets, meta in loader:
            if reset_each_sequence and hasattr(model, 'reset_state') and len(meta) == 1:
                seq = meta[0]['sequence']
                if seq != last_seq:
                    model.reset_state()
                    last_seq = seq
            imgs = [x.to(device) for x in images]
            preds = model(imgs)
            preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
            tgts_cpu = [{k: v.detach().cpu() for k, v in t.items() if k in ('boxes','labels')} for t in targets]
            metric.update(preds_cpu, tgts_cpu)
    return metric.compute()
