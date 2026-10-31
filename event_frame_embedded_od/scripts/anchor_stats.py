#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from efod.config import DSEC_CLASSES


def existing(seq, rels):
    for r in rels:
        p = seq / r
        if p.exists(): return p
    return None


def field(names, options):
    for x in options:
        if x in names: return x
    raise KeyError((names, options))


def main():
    p=argparse.ArgumentParser(description='Training-only DSEC box scale/aspect statistics for SSD anchor selection')
    p.add_argument('--root',required=True); p.add_argument('--split',default='train')
    p.add_argument('--width',type=int,default=640); p.add_argument('--height',type=int,default=480)
    p.add_argument('--csv-out',default='dsec_anchor_stats.csv'); a=p.parse_args()
    root=Path(a.root); base=root/a.split if (root/a.split).exists() else root
    scales=[]; aspects=[]; classes=[]; rows=[]
    for seq in sorted(x for x in base.iterdir() if x.is_dir()):
        tp=existing(seq,['object_detections/left/tracks_remapped.npy','object_detections/left/tracks.npy'])
        tsp=existing(seq,['images/timestamps.txt'])
        if tp is None or tsp is None: continue
        tr=np.load(tp,mmap_mode='r'); names=tr.dtype.names or (); tf=field(names,['t','timestamp','ts']); wf=field(names,['w','width']); hf=field(names,['h','height']); cf=field(names,['class_id','class','label'])
        valid_ts=np.atleast_1d(np.loadtxt(tsp,dtype=np.int64)); valid_ts_set=set(map(int,valid_ts.tolist()))
        # Restrict to annotations associated with available image timestamps. This avoids using labels outside a downloaded fragment/timeline.
        mask=np.fromiter((int(t) in valid_ts_set for t in tr[tf]),dtype=bool,count=len(tr))
        x=tr[mask]; w=x[wf].astype(np.float64); h=x[hf].astype(np.float64); c=x[cf].astype(np.int64)
        valid=(w>0)&(h>0)&(c>=0)&(c<len(DSEC_CLASSES)); w=w[valid]; h=h[valid]; c=c[valid]
        sc=np.sqrt((w*h)/(a.width*a.height)); ar=w/h
        scales.extend(sc.tolist()); aspects.extend(ar.tolist()); classes.extend(c.tolist())
        rows.append({'sequence':seq.name,'boxes_used':int(len(sc))})
    if not scales: raise SystemExit('No usable boxes found')
    s=np.asarray(scales); ar=np.asarray(aspects); c=np.asarray(classes)
    q=[1,5,10,25,50,75,90,95,99]
    print('boxes used:',len(s))
    print('normalized sqrt-area scale percentiles:')
    for x in q: print(f'  p{x:02d}: {np.percentile(s,x):.6f}')
    print('aspect-ratio percentiles:')
    for x in q: print(f'  p{x:02d}: {np.percentile(ar,x):.4f}')
    print('\nclass counts:')
    for i,n in enumerate(DSEC_CLASSES): print(f'  {n:12s}: {int((c==i).sum())}')
    print(f'\nSuggested exploratory min_ratio≈p5={np.percentile(s,5):.4f}, max_ratio≈p95={np.percentile(s,95):.4f}')
    pd.DataFrame(rows).to_csv(a.csv_out,index=False); print('saved:',a.csv_out)

if __name__=='__main__': main()
