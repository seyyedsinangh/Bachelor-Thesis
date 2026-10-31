#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from efod import build_model, MODEL_SPECS
from efod.data import DSECAlignedDataset, detection_collate
from efod.training import (
    make_optimizer_scheduler,
    train_snn_no_reset_epoch,
    save_checkpoint,
    probe_snn_activity,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Limited-data no-reset training of the six Event-SNN detectors on one DSEC/DAGr sequence.'
    )
    p.add_argument('--sequence-dir', required=True, help='Path to one DSEC sequence, e.g. .../zurich_city_13_b')
    p.add_argument('--variants', nargs='+', default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    p.add_argument('--epochs', type=int, default=10, help='Fixed epoch budget; no validation/best-epoch selection is used.')
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--max-unlabeled-steps', type=int, default=5)
    p.add_argument('--window-us', type=int, default=50_000)
    p.add_argument('--event-direction', choices=['past', 'future'], default='past')
    p.add_argument('--workers', type=int, default=0, help='0 is safest for HDF5 in Colab.')
    p.add_argument('--probe-windows', type=int, default=32)
    p.add_argument('--probe-start-fraction', type=float, default=.25)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--outdir', default='fragment_checkpoints')
    p.add_argument('--save-optimizer-state', action='store_true', help='Store Adam/scheduler state too; off by default to minimize checkpoint storage.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_from_sequence(seq: Path, window_us: int, direction: str):
    if not (seq / 'events/left/events.h5').exists():
        raise FileNotFoundError(seq / 'events/left/events.h5')
    return DSECAlignedDataset(
        root=seq.parent,
        split=None,
        domain='event',
        window_us=window_us,
        event_direction=direction,
        sequence_names=[seq.name],
    )


def labeled_count(ds):
    n = 0
    boxes = 0
    for r in ds.records:
        tracks = ds._tracks(r.tracks_path)
        from efod.data.dsec import tracks_to_target
        t = tracks_to_target(tracks, r.timestamp_us, ds.width, ds.height)
        if t['boxes'].numel():
            n += 1
            boxes += int(t['boxes'].shape[0])
    return n, boxes


def main():
    a = parse_args()
    seq = Path(a.sequence_dir).resolve()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / 'training_history.csv'
    summary_path = out / 'training_summary.json'

    ds = dataset_from_sequence(seq, a.window_us, a.event_direction)
    n_labeled, n_boxes = labeled_count(ds)
    summary = {
        'sequence_dir': str(seq),
        'sequence': seq.name,
        'records': len(ds),
        'labeled_records': n_labeled,
        'boxes': n_boxes,
        'window_us': a.window_us,
        'event_direction': a.event_direction,
        'epochs': a.epochs,
        'lr': a.lr,
        'weight_decay': a.weight_decay,
        'grad_clip': a.grad_clip,
        'max_unlabeled_steps': a.max_unlabeled_steps,
        'variants': a.variants,
        'seed': a.seed,
        'save_optimizer_state': a.save_optimizer_state,
        'purpose': 'limited-data in-sample SNN activity/computation characterization; not a DSEC generalization benchmark',
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if n_labeled == 0:
        raise SystemExit('No labeled windows found; detector training requires tracks.npy labels.')

    rows = []
    fields = [
        'variant', 'epoch', 'loss', 'bbox_regression_loss', 'classification_loss', 'updates', 'lr',
        'probe_spike_rate', 'probe_spike_rate_pct', 'probe_spike_sparsity', 'probe_windows', 'checkpoint'
    ]

    for vi, variant in enumerate(a.variants):
        print(f'\n===== TRAIN {variant} ({vi+1}/{len(a.variants)}) =====', flush=True)
        seed_everything(a.seed)
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=a.workers,
            collate_fn=detection_collate, pin_memory=str(a.device).startswith('cuda')
        )
        model = build_model(variant, 'event_snn').to(a.device)
        opt, sch = make_optimizer_scheduler(model, a.epochs, a.lr, a.weight_decay)
        checkpoint = out / f'{variant}_event_snn.pt'

        pre = probe_snn_activity(
            model, ds, a.device, max_windows=a.probe_windows,
            start_fraction=a.probe_start_fraction
        )
        print(f"initial eval-mode probe activity: {100*pre['probe_spike_rate']:.4f}% over {pre['probe_windows']} windows")

        for epoch in range(a.epochs):
            stats = train_snn_no_reset_epoch(
                model, loader, opt, a.device,
                max_unlabeled_steps=a.max_unlabeled_steps,
                grad_clip=a.grad_clip,
            )
            sch.step()
            probe = probe_snn_activity(
                model, ds, a.device, max_windows=a.probe_windows,
                start_fraction=a.probe_start_fraction
            )
            row = {
                'variant': variant,
                'epoch': epoch,
                **stats,
                'lr': sch.get_last_lr()[0],
                'probe_spike_rate': probe['probe_spike_rate'],
                'probe_spike_rate_pct': 100 * probe['probe_spike_rate'],
                'probe_spike_sparsity': probe['probe_spike_sparsity'],
                'probe_windows': probe['probe_windows'],
                'checkpoint': str(checkpoint),
            }
            rows.append(row)
            save_checkpoint(
                checkpoint, model, opt if a.save_optimizer_state else None, sch if a.save_optimizer_state else None, epoch,
                extra={
                    'train_stats': stats,
                    'probe_activity': probe,
                    'limited_fragment_training': True,
                    'sequence': seq.name,
                    'uses_all_available_sequence_for_training': True,
                    'not_generalization_evaluation': True,
                },
            )
            with history_path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(
                f"epoch={epoch:03d} loss={stats['loss']:.6g} "
                f"bbox={stats['bbox_regression_loss']:.5g} cls={stats['classification_loss']:.5g} "
                f"updates={stats['updates']} activity={100*probe['probe_spike_rate']:.3f}% "
                f"lr={sch.get_last_lr()[0]:.3g}",
                flush=True,
            )

        print('saved:', checkpoint.resolve())
        del model, opt, sch, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\nTraining complete.')
    print('History:', history_path.resolve())
    print('Checkpoints:', out.resolve())


if __name__ == '__main__':
    main()
