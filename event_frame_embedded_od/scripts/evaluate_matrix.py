#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from efod import build_model,MODEL_SPECS,PIPELINES
from efod.data import DSECAlignedDataset,detection_collate
from efod.training import load_checkpoint
from efod.evaluation import evaluate_map


def scalar(v):
    if torch.is_tensor(v):
        if v.numel()==1: return float(v)
        return json.dumps(v.tolist())
    return v


def main():
    p=argparse.ArgumentParser(description='Evaluate all available A/B/C checkpoints with COCO mAP')
    p.add_argument('--root',required=True); p.add_argument('--split',default='val'); p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--workers',type=int,default=2); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--out',default='evaluation_matrix.csv'); a=p.parse_args()
    rows=[]; ckroot=Path(a.checkpoint_dir)
    for variant in MODEL_SPECS:
        for pipeline in PIPELINES:
            ck=ckroot/f'{variant}_{pipeline}.pt'
            if not ck.exists():
                rows.append({'variant':variant,'pipeline':pipeline,'checkpoint':str(ck),'status':'missing'}); continue
            domain='event' if pipeline=='event_snn' else 'frame'
            ds=DSECAlignedDataset(a.root,a.split,domain); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=a.workers,collate_fn=detection_collate)
            m=build_model(variant,pipeline).to(a.device); load_checkpoint(ck,m,map_location=a.device)
            r=evaluate_map(m,loader,a.device,reset_each_sequence=True)
            row={'variant':variant,'pipeline':pipeline,'checkpoint':str(ck),'status':'ok'}
            row.update({k:scalar(v) for k,v in r.items()}); rows.append(row); print(variant,pipeline,row.get('map'))
    pd.DataFrame(rows).to_csv(a.out,index=False); print('saved:',a.out)

if __name__=='__main__': main()
