#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

from efod import build_model, MODEL_SPECS, PIPELINES, DSEC_CLASSES
from efod.training import load_checkpoint
from efod.data.dsec import load_event_window, load_rgb_frame, tracks_to_target


def find_demo(seq: Path):
    ts = np.atleast_1d(np.loadtxt(seq / 'images/timestamps.txt', dtype=np.int64))
    imgs = sorted((seq / 'images/left/distorted').glob('*.png'))
    tracks_path = seq / 'object_detections/left/tracks.npy'
    tracks = np.load(tracks_path)
    n = min(len(ts), len(imgs))
    for i in range(1, n):
        target = tracks_to_target(tracks, int(ts[i]), 640, 480)
        if target['boxes'].shape[0] > 0:
            return i, int(ts[i]), imgs[i], target
    i = min(1, n-1)
    return i, int(ts[i]), imgs[i], tracks_to_target(tracks, int(ts[i]), 640, 480)


def event_canvas(event):
    e = event.detach().cpu().numpy()
    rgb = np.zeros((480,640,3), dtype=np.uint8)
    rgb[...,0] = (e[1] > 0).astype(np.uint8) * 255
    rgb[...,2] = (e[0] > 0).astype(np.uint8) * 255
    return Image.fromarray(rgb)


def draw_boxes(base: Image.Image, target, pred, topk=5):
    im = base.copy(); d = ImageDraw.Draw(im)
    for box, lab in zip(target['boxes'].tolist(), target['labels'].tolist()):
        d.rectangle(box, outline='green', width=3)
        name = DSEC_CLASSES[lab] if 0 <= lab < len(DSEC_CLASSES) else str(lab)
        d.text((box[0]+2, box[1]+2), f'GT {name}', fill='green')
    order = torch.argsort(pred['scores'], descending=True)[:topk]
    for j in order.tolist():
        box = pred['boxes'][j].detach().cpu().tolist()
        lab = int(pred['labels'][j])
        score = float(pred['scores'][j])
        d.rectangle(box, outline='yellow', width=2)
        name = DSEC_CLASSES[lab] if 0 <= lab < len(DSEC_CLASSES) else str(lab)
        d.text((box[0]+2, max(0, box[1]-12)), f'{name} {score:.2f}', fill='yellow')
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence-dir', required=True)
    ap.add_argument('--outdir', default='results/fun_detections')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--checkpoint-dir', default=None, help='Optional {variant}_{pipeline}.pt checkpoint folder')
    a = ap.parse_args()
    seq = Path(a.sequence_dir); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    i, t, frame_path, target = find_demo(seq)
    event = load_event_window(seq/'events/left/events.h5', t, direction='past').to(a.device)
    frame = load_rgb_frame(frame_path).to(a.device)
    frame_pil = Image.open(frame_path).convert('RGB')

    for variant in MODEL_SPECS:
        for pipeline in PIPELINES:
            model = build_model(variant, pipeline).to(a.device).eval()
            if a.checkpoint_dir:
                ckpt=Path(a.checkpoint_dir)/f'{variant}_{pipeline}.pt'
                if ckpt.exists(): load_checkpoint(ckpt,model,map_location=a.device)
            model.reset_state()
            x = event if pipeline == 'event_snn' else frame
            with torch.no_grad():
                pred = model([x])[0]
            base = event_canvas(event) if pipeline == 'event_snn' else frame_pil
            vis = draw_boxes(base, target, pred)
            path = out / f'{variant}_{pipeline}.png'
            vis.save(path)
            print(path, 'top score=', float(pred['scores'].max()) if pred['scores'].numel() else None)

if __name__ == '__main__':
    main()
