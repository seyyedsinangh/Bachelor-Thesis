#!/usr/bin/env python
from __future__ import annotations
import argparse,json,torch
from torch.utils.data import DataLoader
from efod import MODEL_SPECS,PIPELINES
from efod.data import DSECAlignedDataset,detection_collate
from efod.training import load_model_from_checkpoint
from efod.evaluation import evaluate_map

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--split',default='val'); p.add_argument('--checkpoint',required=True)
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--workers',type=int,default=2); p.add_argument('--window-us',type=int,default=50_000); p.add_argument('--event-direction',choices=['past','future'],default='past'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--json-out',default=None); a=p.parse_args()
    model,ck=load_model_from_checkpoint(a.checkpoint,a.device); pipeline=model.cfg.pipeline; domain='event' if pipeline=='event_snn' else 'frame'
    if pipeline=='event_snn' and a.batch_size!=1: print('warning: batch_size=1 is recommended for stateful sequential event evaluation')
    ds=DSECAlignedDataset(a.root,a.split,domain,a.window_us,a.event_direction); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,collate_fn=detection_collate)
    r=evaluate_map(model,loader,a.device,reset_each_sequence=True); serial={k:(v.tolist() if torch.is_tensor(v) else v) for k,v in r.items()}; serial['checkpoint_config']=ck['config']; print(json.dumps(serial,indent=2))
    if a.json_out: open(a.json_out,'w').write(json.dumps(serial,indent=2))
if __name__=='__main__': main()
