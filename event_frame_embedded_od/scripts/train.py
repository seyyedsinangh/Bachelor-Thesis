#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from efod import build_model,MODEL_SPECS,PIPELINES
from efod.data import DSECAlignedDataset,detection_collate
from efod.training import make_optimizer_scheduler,train_ann_epoch,train_snn_reset_epoch,train_snn_no_reset_epoch,save_checkpoint,load_checkpoint

def args():
    p=argparse.ArgumentParser(description='Train event-SNN or RGB-ANN DSEC detector')
    p.add_argument('--root',required=True); p.add_argument('--split',default='train'); p.add_argument('--variant',choices=MODEL_SPECS,default='vgg32'); p.add_argument('--pipeline',choices=PIPELINES,default='event_snn')
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=1); p.add_argument('--workers',type=int,default=2); p.add_argument('--lr',type=float,default=5e-3); p.add_argument('--weight-decay',type=float,default=1e-2)
    p.add_argument('--window-us',type=int,default=50_000); p.add_argument('--event-direction',choices=['past','future'],default='past'); p.add_argument('--snn-mode',choices=['no_reset','reset'],default='no_reset')
    p.add_argument('--max-unlabeled-steps',type=int,default=5); p.add_argument('--min-ratio',type=float,default=0.015); p.add_argument('--max-ratio',type=float,default=0.25); p.add_argument('--grad-clip',type=float,default=None); p.add_argument('--out',default='checkpoints/model.pt'); p.add_argument('--resume',default=None)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); return p.parse_args()

def main():
    a=args(); domain='event' if a.pipeline=='event_snn' else 'frame'; ds=DSECAlignedDataset(a.root,a.split,domain,a.window_us,a.event_direction)
    sequential=a.pipeline=='event_snn' and a.snn_mode=='no_reset';
    if sequential and a.batch_size!=1: raise SystemExit('event_snn --snn-mode no_reset requires --batch-size 1')
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=not sequential,num_workers=a.workers,collate_fn=detection_collate,pin_memory=a.device.startswith('cuda'))
    model=build_model(a.variant,a.pipeline,min_ratio=a.min_ratio,max_ratio=a.max_ratio).to(a.device); opt,sch=make_optimizer_scheduler(model,a.epochs,a.lr,a.weight_decay); start=0
    if a.resume:
        ck=load_checkpoint(a.resume,model,opt,sch,map_location=a.device); start=int(ck.get('epoch') or -1)+1; print('resumed at epoch',start)
    for epoch in range(start,a.epochs):
        if a.pipeline!='event_snn': stats=train_ann_epoch(model,loader,opt,a.device,a.grad_clip)
        elif a.snn_mode=='no_reset': stats=train_snn_no_reset_epoch(model,loader,opt,a.device,a.max_unlabeled_steps,a.grad_clip)
        else: stats=train_snn_reset_epoch(model,loader,opt,a.device,a.grad_clip)
        sch.step(); print(f"epoch={epoch:03d} loss={stats['loss']:.6g} bbox={stats['bbox_regression_loss']:.5g} cls={stats['classification_loss']:.5g} updates={stats['updates']} lr={sch.get_last_lr()[0]:.3g}")
        save_checkpoint(a.out,model,opt,sch,epoch,{'train_stats':stats,'snn_mode':a.snn_mode})
    print('saved:',Path(a.out).resolve())
if __name__=='__main__': main()
