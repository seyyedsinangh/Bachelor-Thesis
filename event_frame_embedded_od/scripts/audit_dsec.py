#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import hdf5plugin
import numpy as np

hdf5plugin.register(force=True)
import pandas as pd


def first_existing(seq: Path, rels):
    for rel in rels:
        p = seq / rel
        if p.exists():
            return p
    return None


def inspect_sequence(seq: Path):
    events = first_existing(seq, ['events/left/events.h5'])
    tracks = first_existing(seq, [
        'object_detections/left/tracks_remapped.npy',
        'object_detections/left/tracks.npy',
    ])
    tsp = first_existing(seq, ['images/timestamps.txt'])
    image_dir = first_existing(seq, ['images/left/rectified','images/left/distorted','images/left'])
    row = {'sequence': seq.name, 'ok': False}
    if not all(x is not None for x in (events, tracks, tsp, image_dir)):
        row['error'] = 'missing events/tracks/timestamps/images'
        return row
    try:
        ts = np.atleast_1d(np.loadtxt(tsp, dtype=np.int64))
        imgs = sorted(image_dir.glob('*.png'))
        tr = np.load(tracks, mmap_mode='r')
        tf = 't' if tr.dtype.names and 't' in tr.dtype.names else None
        with h5py.File(events, 'r') as f:
            t0r = int(f['events/t'][0]); t1r = int(f['events/t'][-1])
            off = int(f['t_offset'][()]) if 't_offset' in f else 0
        t0a, t1a = off + t0r, off + t1r
        label_min = int(tr[tf].min()) if tf and len(tr) else None
        label_max = int(tr[tf].max()) if tf and len(tr) else None
        n_images = len(imgs); n_ts = len(ts)
        usable = min(n_images, n_ts)
        label_overlap = None if label_min is None else max(0, min(label_max, t1a) - max(label_min, t0a))
        row.update({
            'ok': bool(usable > 0 and t1a > t0a),
            'images': n_images,
            'timestamps': n_ts,
            'usable_frame_records': usable,
            'event_abs_start_us': t0a,
            'event_abs_end_us': t1a,
            'event_duration_s': (t1a - t0a) / 1e6,
            'label_rows': int(len(tr)),
            'label_min_us': label_min,
            'label_max_us': label_max,
            'label_event_overlap_us': label_overlap,
            'image_ts_min_us': int(ts.min()) if len(ts) else None,
            'image_ts_max_us': int(ts.max()) if len(ts) else None,
            'image_timestamp_count_matches_png': n_images == n_ts,
        })
    except Exception as e:
        row['error'] = repr(e)
    return row


def main():
    p = argparse.ArgumentParser(description='Audit DSEC sequence structure/timelines before training')
    p.add_argument('--root', required=True)
    p.add_argument('--split', default=None)
    p.add_argument('--csv-out', default='dsec_audit.csv')
    a = p.parse_args()
    root = Path(a.root)
    base = root / a.split if a.split and (root / a.split).exists() else root
    seqs = sorted(x for x in base.iterdir() if x.is_dir())
    rows = [inspect_sequence(s) for s in seqs]
    df = pd.DataFrame(rows)
    df.to_csv(a.csv_out, index=False)
    print(df.to_string(index=False))
    bad = df[~df['ok'].fillna(False)] if len(df) else df
    print(f'\nSequences: {len(df)} | PASS: {len(df)-len(bad)} | FAIL: {len(bad)}')
    print('saved:', a.csv_out)
    if len(bad):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
