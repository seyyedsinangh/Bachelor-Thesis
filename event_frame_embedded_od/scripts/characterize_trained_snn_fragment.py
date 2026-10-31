#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from efod import build_model, MODEL_SPECS
from efod.data import DSECAlignedDataset, detection_collate
from efod.training import load_checkpoint
from efod.evaluation import parameter_summary, dense_conv_macs, membrane_state_mib
from efod.evaluation.complexity import spike_activity
from efod.evaluation.thesis_energy import estimate_thesis_energy
from efod.evaluation.gpu_energy import measure_gpu_energy_sequence


def parse_args():
    p = argparse.ArgumentParser(description='Streaming in-sample activity/energy characterization of trained Event-SNNs.')
    p.add_argument('--sequence-dir', required=True)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--variants', nargs='+', default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    p.add_argument('--window-us', type=int, default=50_000)
    p.add_argument('--event-direction', choices=['past', 'future'], default='past')
    p.add_argument('--max-windows', type=int, default=0, help='0 = all available windows.')
    p.add_argument('--timing-windows', type=int, default=40)
    p.add_argument('--gpu-energy-windows', type=int, default=12)
    p.add_argument('--gpu-energy-seconds', type=float, default=2.0)
    p.add_argument('--gpu-energy', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--outdir', default='trained_fragment_results')
    return p.parse_args()


def make_dataset(seq: Path, window_us: int, direction: str):
    return DSECAlignedDataset(
        root=seq.parent, split=None, domain='event', window_us=window_us,
        event_direction=direction, sequence_names=[seq.name]
    )


def _pctile95(xs):
    if not xs:
        return float('nan')
    s = sorted(xs)
    return s[min(len(s)-1, int(round(.95*(len(s)-1))))]


def _time_forward(model, ds, device, max_windows):
    n = min(len(ds), max_windows) if max_windows > 0 else len(ds)
    if n == 0:
        return {}
    # Use a contiguous chronological segment; data loading and host->device copy are outside the timer.
    start = max(0, (len(ds)-n)//2)
    times = []
    model.eval(); model.reset_state()
    with torch.no_grad():
        for j in range(start, start+n):
            image, _, _ = ds[j]
            x = image.unsqueeze(0).to(device)
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
                a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                a.record(); model.forward_raw(x); b.record(); torch.cuda.synchronize()
                times.append(float(a.elapsed_time(b)))
            else:
                t0 = time.perf_counter(); model.forward_raw(x); times.append((time.perf_counter()-t0)*1000)
    mean = statistics.fmean(times)
    return {
        'trained_stream_latency_mean_ms': mean,
        'trained_stream_latency_median_ms': statistics.median(times),
        'trained_stream_latency_p95_ms': _pctile95(times),
        'trained_stream_fps': 1000/mean if mean > 0 else float('inf'),
        'trained_stream_timing_windows': len(times),
    }


def _gpu_sample_inputs(ds, device, n):
    n = min(max(1, n), len(ds))
    start = max(0, (len(ds)-n)//2)
    xs = []
    for j in range(start, start+n):
        image, _, _ = ds[j]
        xs.append(image.unsqueeze(0).to(device))
    return xs


def main():
    a = parse_args()
    seq = Path(a.sequence_dir).resolve(); ckdir = Path(a.checkpoint_dir).resolve(); out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True); (out/'layer_activity').mkdir(exist_ok=True); (out/'layer_energy').mkdir(exist_ok=True)
    ds = make_dataset(seq, a.window_us, a.event_direction)
    n_total = len(ds) if a.max_windows <= 0 else min(len(ds), a.max_windows)
    print(f'Sequence={seq.name} windows={n_total}/{len(ds)} device={a.device}')
    rows = []

    for variant in a.variants:
        ckpt = ckdir / f'{variant}_event_snn.pt'
        if not ckpt.exists():
            print('SKIP missing', ckpt); continue
        print(f'\n===== CHARACTERIZE {variant} =====', flush=True)
        model = build_model(variant, 'event_snn').to(a.device).eval()
        load_checkpoint(ckpt, model, map_location=a.device)
        model.reset_state()

        energy_sums = {}
        layer_spikes = {}; layer_energy = {}
        total_spikes = total_neurons = 0
        input_nonzero = input_total = 0
        used = 0
        first_x = None

        for idx in range(n_total):
            image, _, _ = ds[idx]
            input_nonzero += int((image != 0).count_nonzero().item()); input_total += image.numel()
            x = image.unsqueeze(0).to(a.device)
            if first_x is None:
                first_x = x
            # This performs the one chronological state-advancing forward for the analytical pass.
            e = estimate_thesis_energy(model, x, reset_state=False)
            erows = e.pop('layer_rows')
            act = spike_activity(model)
            total_spikes += int(act['spikes']); total_neurons += int(act['spike_neurons_observed'])
            for lr in act['spike_layer_rows']:
                rec = layer_spikes.setdefault(lr['name'], [0,0])
                rec[0] += int(lr['spikes']); rec[1] += int(lr['neurons'])
            for lr in erows:
                rec = layer_energy.setdefault(lr['name'], {'kind':lr['kind'],'windows':0})
                rec['windows'] += 1
                for k in ('synaptic_macs','synaptic_accs','address_macs','address_accs','in_reads','parameter_reads','potential_reads','output_writes','potential_writes','memory_energy_pj','synaptic_energy_pj','address_energy_pj','total_energy_pj'):
                    rec[k] = rec.get(k, 0.0) + float(lr[k])
            for k,v in e.items():
                if isinstance(v,(int,float)) and math.isfinite(float(v)):
                    energy_sums[k] = energy_sums.get(k,0.0) + float(v)
            used += 1
            if used % 100 == 0:
                print(f'  analytical/activity windows: {used}/{n_total}', flush=True)

        rate = total_spikes/max(1,total_neurons)
        row = {
            'variant': variant,
            'model': MODEL_SPECS[variant]['display'],
            'weights_state': 'trained_limited_fragment',
            'checkpoint': str(ckpt),
            'sequence': seq.name,
            'characterization_windows': used,
            'in_sample_characterization': True,
            'generalization_claim_valid': False,
            'event_input_occupancy': input_nonzero/max(1,input_total),
            'trained_seq_spikes_total': total_spikes,
            'trained_seq_spike_rate': rate,
            'trained_seq_spike_rate_pct': 100*rate,
            'trained_seq_spike_sparsity': 1-rate,
        }
        row.update(parameter_summary(model))
        if first_x is not None:
            dense,_ = dense_conv_macs(model, first_x)
            row['dense_macs'] = dense; row['dense_gmacs'] = dense/1e9
        row.update(membrane_state_mib(model))
        for k,v in energy_sums.items():
            row[k] = v/max(1,used)
        if 'thesis_total_uJ' in row:
            row['thesis_sequence_total_mJ'] = row['thesis_total_uJ']*used/1000.0
        row.update(_time_forward(model, ds, a.device, a.timing_windows))
        if a.gpu_energy and str(a.device).startswith('cuda'):
            xs = _gpu_sample_inputs(ds, a.device, a.gpu_energy_windows)
            row.update(measure_gpu_energy_sequence(model, xs, duration_s=a.gpu_energy_seconds, state_mode='stream'))
            del xs
        try:
            row['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        except Exception:
            row['gpu_name'] = 'unknown'
        rows.append(row)

        la = []
        for name,(sp,neu) in layer_spikes.items():
            r = sp/max(1,neu)
            la.append({'variant':variant,'layer':name,'spikes':sp,'neurons':neu,'spike_rate':r,'spike_rate_pct':100*r,'sparsity':1-r})
        pd.DataFrame(la).to_csv(out/'layer_activity'/f'{variant}.csv',index=False)
        le = []
        for name,rec in layer_energy.items():
            n = rec['windows']; z={'variant':variant,'layer':name,'kind':rec['kind'],'windows':n}
            for k,v in rec.items():
                if k not in ('kind','windows'): z[f'mean_{k}']=v/max(1,n)
            le.append(z)
        pd.DataFrame(le).to_csv(out/'layer_energy'/f'{variant}.csv',index=False)

        print({k:row.get(k) for k in ('trained_seq_spike_rate_pct','thesis_total_uJ','trained_stream_latency_mean_ms','gpu_seq_energy_dynamic_mj')})
        del model, first_x
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out/'trained_snn_characterization.csv',index=False)
    (out/'trained_snn_characterization.md').write_text(df.to_markdown(index=False) if len(df) else 'No checkpoints found.\n')
    meta = {
        'interpretation': 'In-sample computational characterization after limited-data training on the same available DSEC/DAGr fragment. Do not report mAP from this pass as generalization.',
        'sequence': seq.name,
        'windows': n_total,
        'window_us': a.window_us,
        'state_mode': 'stream/no-reset',
    }
    (out/'characterization_meta.json').write_text(json.dumps(meta,indent=2))
    print('\nSaved:', (out/'trained_snn_characterization.csv').resolve())


if __name__ == '__main__':
    main()
