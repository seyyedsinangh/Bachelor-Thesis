#!/usr/bin/env python
from __future__ import annotations

import argparse,json,gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from efod import build_model,MODEL_SPECS,PIPELINES,PIPELINE_DISPLAY
from efod.training import load_checkpoint
from efod.data.dsec import load_event_window,load_rgb_frame
from efod.evaluation import (
    parameter_summary,dense_conv_macs,peak_cuda_memory,membrane_state_mib,
    benchmark_latency_sequence,spike_activity_sequence,estimate_thesis_energy_sequence,
    measure_gpu_energy_sequence,
)


def _image_dir(seq):
    for rel in ('images/left/rectified','images/left/distorted','images/left'):
        p=seq/rel
        if p.exists() and list(p.glob('*.png')): return p
    raise FileNotFoundError('No DSEC left image directory found')

def _event_workload_summary(event_inputs):
    if not event_inputs:
        return {}
    occ=[]; nz=[]; pos=[]; neg=[]; absmean=[]
    for x in event_inputs:
        y=x.detach()
        occ.append(float((y!=0).float().mean().item()))
        nz.append(int((y!=0).sum().item()))
        absmean.append(float(y.abs().mean().item()))
        if y.shape[1] >= 2:
            pos.append(float((y[:,0]!=0).float().mean().item()))
            neg.append(float((y[:,1]!=0).float().mean().item()))
    return {
        'event_input_occupancy_mean':float(np.mean(occ)),
        'event_input_occupancy_std':float(np.std(occ,ddof=1)) if len(occ)>1 else 0.0,
        'event_input_occupancy_min':float(np.min(occ)),
        'event_input_occupancy_max':float(np.max(occ)),
        'event_input_nonzero_entries_mean':float(np.mean(nz)),
        'event_input_abs_mean':float(np.mean(absmean)),
        'event_input_positive_occupancy_mean':float(np.mean(pos)) if pos else float('nan'),
        'event_input_negative_occupancy_mean':float(np.mean(neg)) if neg else float('nan'),
    }

def get_input_sequence(sequence_dir,device,windows=8):
    if sequence_dir is None:
        g=torch.Generator().manual_seed(12345)
        nwin=8 if windows is None or windows<=0 else windows
        ev=[]; fr=[]
        for _ in range(nwin):
            ev.append((torch.rand((1,2,480,640),generator=g)<.01).float().to(device))
            fr.append(torch.rand((1,3,480,640),generator=g).to(device))
        meta={'source':'synthetic','windows':nwin,'available_windows':nwin}
        meta.update(_event_workload_summary(ev))
        return ev,fr,meta
    seq=Path(sequence_dir); ts=np.atleast_1d(np.loadtxt(seq/'images/timestamps.txt',dtype=np.int64)); imgs=sorted(_image_dir(seq).glob('*.png')); n=min(len(ts),len(imgs))
    # Skip the first image so a full causal 50ms past window normally exists.
    all_ids=list(range(1,n)) if n>1 else [0]
    ids=all_ids if windows is None or windows<=0 else all_ids[:windows]
    ev=[load_event_window(seq/'events/left/events.h5',int(ts[i]),direction='past').unsqueeze(0).to(device) for i in ids]
    fr=[load_rgb_frame(imgs[i]).unsqueeze(0).to(device) for i in ids]
    meta={'source':str(seq),'windows':len(ids),'available_windows':len(all_ids),'indices':ids,'timestamps_us':[int(ts[i]) for i in ids],'frames':[str(imgs[i]) for i in ids]}
    meta.update(_event_workload_summary(ev))
    return ev,fr,meta

def _scalarize(d):
    return {k:v for k,v in d.items() if isinstance(v,(str,int,float,bool)) or v is None}

def main():
    p=argparse.ArgumentParser(description='18-model event-SNN/RGB-ANN architecture + energy benchmark')
    p.add_argument('--sequence-dir',default=None,help='DSEC sequence folder; synthetic inputs if omitted')
    p.add_argument('--windows',type=int,default=8,help='Number of chronological windows; <=0 means all available windows')
    p.add_argument('--outdir',default='results'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--fast',action='store_true'); p.add_argument('--gpu-energy',action='store_true')
    p.add_argument('--gpu-energy-repeats',type=int,default=5,help='Repeated NVML active runs (final protocol: 5-10)')
    p.add_argument('--gpu-energy-duration',type=float,default=10.0,help='Active seconds per NVML repeat (final protocol: 10-20 s)')
    p.add_argument('--gpu-idle-duration',type=float,default=2.0,help='Idle power sampling seconds before and after every active repeat')
    p.add_argument('--gpu-poll-interval',type=float,default=0.02,help='NVML power polling interval in seconds')
    p.add_argument('--checkpoint-dir',default=None,help='Folder containing optional {variant}_{pipeline}.pt checkpoints')
    p.add_argument('--require-trained-snn',action='store_true',help='Fail if any Event-SNN checkpoint is missing. Recommended after limited-fragment training.')
    p.add_argument('--seed',type=int,default=12345); a=p.parse_args()
    device=torch.device(a.device); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    for s in ('layer_energy','spike_activity','spike_activity_summary','gpu_energy','architectures'): (out/s).mkdir(exist_ok=True)
    event_inputs,frame_inputs,meta=get_input_sequence(a.sequence_dir,device,a.windows); (out/'input_meta.json').write_text(json.dumps(meta,indent=2))
    rows=[]
    for variant in MODEL_SPECS:
        for pipeline in PIPELINES:
            print(f'\n=== {variant} | {PIPELINE_DISPLAY[pipeline]} ===',flush=True)
            torch.manual_seed(a.seed)
            model=build_model(variant,pipeline).to(device).eval(); weights_state='random_untrained'; checkpoint_used=None
            if a.checkpoint_dir:
                ckpt=Path(a.checkpoint_dir)/f'{variant}_{pipeline}.pt'
                if ckpt.exists():
                    payload=load_checkpoint(ckpt,model,map_location=device)
                    checkpoint_used=str(ckpt)
                    extra=payload.get('extra',{}) if isinstance(payload,dict) else {}
                    weights_state='trained_limited_fragment' if extra.get('limited_fragment_training') else 'trained_checkpoint'
            if pipeline=='event_snn' and a.require_trained_snn and weights_state=='random_untrained':
                expected=Path(a.checkpoint_dir or '.')/f'{variant}_event_snn.pt'
                raise FileNotFoundError(
                    f'Required trained Event-SNN checkpoint missing for {variant}: {expected}. '
                    'Run scripts/train_fragment_snn_matrix.py first.'
                )
            inputs=event_inputs if pipeline=='event_snn' else frame_inputs; x=inputs[0]
            row={'variant':variant,'model':MODEL_SPECS[variant]['display'],'pipeline':pipeline,'pipeline_display':PIPELINE_DISPLAY[pipeline],'weights_state':weights_state,'checkpoint':checkpoint_used,'input_channels':x.shape[1],'input_h':x.shape[-2],'input_w':x.shape[-1],'benchmark_windows':len(inputs),'available_windows':meta.get('available_windows',len(inputs))}
            if pipeline=='event_snn':
                for k,v in meta.items():
                    if k.startswith('event_input_'):
                        row[k]=v
            row.update(parameter_summary(model)); dense,convrows=dense_conv_macs(model,x); row.update(dense_macs=dense,dense_gmacs=dense/1e9,dense_gflops_2xmac=2*dense/1e9)
            # Both modes are reported; streaming is the meaningful Event-SNN real-time mode.
            row.update(benchmark_latency_sequence(model,inputs,warmup=1 if a.fast else 3,repeats=2 if a.fast else 5,state_mode='stream'))
            row.update(benchmark_latency_sequence(model,inputs,warmup=1 if a.fast else 3,repeats=2 if a.fast else 5,state_mode='reset'))
            row.update(peak_cuda_memory(model,x,state_mode='stream'))
            thesis=estimate_thesis_energy_sequence(model,inputs,state_mode='stream' if pipeline=='event_snn' else 'reset'); layerseq=thesis.pop('layer_rows_sequence'); row.update(_scalarize(thesis))
            # Save per-layer energy averaged across sampled windows.
            if layerseq:
                flat=[]
                for wi,layers in enumerate(layerseq):
                    for r in layers: flat.append({'window':wi,**r})
                pd.DataFrame(flat).to_csv(out/'layer_energy'/f'{variant}_{pipeline}.csv',index=False)
            act=spike_activity_sequence(model,inputs,state_mode='stream' if pipeline=='event_snn' else 'reset'); actrows=act.pop('spike_sequence_rows'); layer_summary=act.pop('spike_layer_summary_rows',[]); row.update(_scalarize(act))
            if pipeline=='event_snn' and weights_state=='random_untrained':
                row['activity_warning']='Spike activity and activity-dependent energy from random weights are structural smoke-test values, not representative trained-network estimates.'
            elif pipeline=='event_snn':
                row['activity_warning']='Learned limited-fragment activity: suitable for in-sample computational characterization, not a held-out DSEC accuracy/generalization claim.'
            if actrows:
                flat=[]
                for wi,r in enumerate(actrows):
                    for lr in r.get('spike_layer_rows',[]): flat.append({'window':wi,**lr})
                pd.DataFrame(flat).to_csv(out/'spike_activity'/f'{variant}_{pipeline}.csv',index=False)
            if layer_summary:
                pd.DataFrame(layer_summary).to_csv(out/'spike_activity_summary'/f'{variant}_{pipeline}.csv',index=False)
            row.update(membrane_state_mib(model))
            # Effective operation fields from the Chapter-6 accounting.
            row['effective_macs_ch6']=row.get('ann_synaptic_macs',0)+row.get('snn_synaptic_macs',0)
            row['effective_accs_ch6']=row.get('ann_synaptic_accs',0)+row.get('snn_synaptic_accs',0)
            if a.gpu_energy:
                energy=measure_gpu_energy_sequence(
                    model,inputs,
                    duration_s=.5 if a.fast else a.gpu_energy_duration,
                    idle_s=.25 if a.fast else a.gpu_idle_duration,
                    poll_interval=a.gpu_poll_interval,
                    state_mode='stream' if pipeline=='event_snn' else 'reset',
                    repeats=2 if a.fast else a.gpu_energy_repeats,
                    warmup_traversals=1 if a.fast else 3,
                    settle_s=.1 if a.fast else .5,
                )
                repeat_rows=energy.pop('gpu_seq_repeat_rows',[])
                row.update(_scalarize(energy))
                if repeat_rows:
                    pd.DataFrame(repeat_rows).to_csv(out/'gpu_energy'/f'{variant}_{pipeline}_repeats.csv',index=False)
            (out/'architectures'/f'{variant}_{pipeline}.txt').write_text(str(model))
            pd.DataFrame(convrows).to_csv(out/'architectures'/f'{variant}_{pipeline}_conv_macs.csv',index=False)
            print({k:row.get(k) for k in ('params','dense_gmacs','seq_latency_stream_mean_ms','seq_spike_rate','thesis_total_uJ','gpu_seq_energy_dynamic_mj')})
            rows.append(row); del model; gc.collect();
            if device.type=='cuda': torch.cuda.empty_cache()
    df=pd.DataFrame(rows); df.to_csv(out/'benchmark_18_models.csv',index=False); (out/'benchmark_18_models.md').write_text(df.to_markdown(index=False))
    health_cols=[c for c in ['variant','model','pipeline','weights_state','seq_spike_rate','seq_spike_layer_count','seq_spike_layer_rate_min','seq_spike_layer_rate_median','seq_spike_layer_rate_max','seq_spike_dead_layers','seq_spike_near_dead_layers','seq_spike_saturated_layers','seq_spike_health_status','event_input_occupancy_mean'] if c in df.columns]
    if health_cols:
        hdf=df[df.pipeline=='event_snn'][health_cols].copy(); hdf.to_csv(out/'snn_spike_health_summary.csv',index=False); (out/'snn_spike_health_summary.md').write_text(hdf.to_markdown(index=False))
    pair=[]
    for v in MODEL_SPECS:
        sub=df[df.variant==v].set_index('pipeline'); A,B,C=sub.loc['event_snn'],sub.loc['strict_ann'],sub.loc['native_ann']
        def ratio(a,b): return float(a/b) if b==b and b!=0 else float('nan')
        pair.append({'variant':v,
          'params_A_over_B':ratio(A.params,B.params),'params_A_over_C':ratio(A.params,C.params),
          'dense_macs_A_over_B':ratio(A.dense_macs,B.dense_macs),'dense_macs_A_over_C':ratio(A.dense_macs,C.dense_macs),
          'effective_macs_A_over_B':ratio(A.effective_macs_ch6,B.effective_macs_ch6),'effective_accs_A_over_B':ratio(A.effective_accs_ch6,B.effective_accs_ch6),
          'latency_stream_A_over_B':ratio(A.seq_latency_stream_mean_ms,B.seq_latency_stream_mean_ms),'latency_stream_A_over_C':ratio(A.seq_latency_stream_mean_ms,C.seq_latency_stream_mean_ms),
          'thesis_energy_A_over_B':ratio(A.thesis_total_uJ,B.thesis_total_uJ),'thesis_energy_A_over_C':ratio(A.thesis_total_uJ,C.thesis_total_uJ),
          'gpu_dynamic_energy_A_over_B':ratio(A.get('gpu_seq_energy_dynamic_mj',float('nan')),B.get('gpu_seq_energy_dynamic_mj',float('nan'))),
          'gpu_dynamic_energy_A_over_C':ratio(A.get('gpu_seq_energy_dynamic_mj',float('nan')),C.get('gpu_seq_energy_dynamic_mj',float('nan'))),
        })
    pdf=pd.DataFrame(pair); pdf.to_csv(out/'paired_ABC_ratios.csv',index=False); (out/'paired_ABC_ratios.md').write_text(pdf.to_markdown(index=False)); print('\nSaved',out/'benchmark_18_models.csv')
if __name__=='__main__': main()
