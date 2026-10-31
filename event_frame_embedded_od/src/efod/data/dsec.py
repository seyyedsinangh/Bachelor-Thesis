from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import bisect

import h5py
import numpy as np

# DSEC event files use HDF5 compression filters that are not part of stock HDF5.
# Importing hdf5plugin registers its embedded filters with the HDF5 library used
# by h5py.  Registration is repeated defensively before reads because notebook
# environments can reload h5py or alter HDF5 plugin search paths.
try:
    import hdf5plugin
except ImportError:  # allow architecture-only use without the optional HDF5 filter package
    hdf5plugin = None

def _ensure_hdf5_filters() -> None:
    if hdf5plugin is None:
        raise ImportError(
            "Reading DSEC events.h5 requires hdf5plugin. Install it with "
            "`python -m pip install hdf5plugin`."
        )
    try:
        hdf5plugin.register(force=True)
    except TypeError:
        # Compatibility with older hdf5plugin versions whose register signature
        # did not expose the force keyword. Importing the module already registers
        # its bundled filters, so this is sufficient.
        hdf5plugin.register()

# Register immediately when available, but do not make architecture-only imports fail.
if hdf5plugin is not None:
    _ensure_hdf5_filters()
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import DSEC_CLASSES


@dataclass(frozen=True)
class DSECRecord:
    sequence: str
    frame_path: str
    timestamp_us: int
    tracks_path: str
    events_path: str


def _sequence_dirs(root: Path, split: str | None) -> list[Path]:
    if split is None:
        base = root
    elif (root / split).is_dir():
        base = root / split
    elif root.name == split:
        base = root
    else:
        raise FileNotFoundError(f'Requested DSEC split {split!r} not found under {root}. Pass split=None if root already contains sequence folders.')
    return sorted(p for p in base.iterdir() if p.is_dir())


def _choose_existing(seq: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = seq / rel
        if p.exists():
            return p
    raise FileNotFoundError(f'None of {candidates} exists under {seq}')


def build_dsec_manifest(
    root: str | Path,
    split: str | None = None,
    sequence_names: tuple[str, ...] | list[str] | None = None,
    image_subdirs: tuple[str, ...] = (
        'images/left/rectified', 'images/left/distorted', 'images/left'
    ),
    tracks_names: tuple[str, ...] = (
        'object_detections/left/tracks_remapped.npy',
        'object_detections/left/tracks.npy',
    ),
) -> list[DSECRecord]:
    root = Path(root)
    records: list[DSECRecord] = []
    wanted = set(sequence_names) if sequence_names is not None else None
    for seq in _sequence_dirs(root, split):
        if wanted is not None and seq.name not in wanted:
            continue
        events = _choose_existing(seq, ['events/left/events.h5'])
        tracks = _choose_existing(seq, list(tracks_names))
        ts_path = _choose_existing(seq, ['images/timestamps.txt'])
        image_dir = _choose_existing(seq, list(image_subdirs))
        timestamps = np.atleast_1d(np.loadtxt(ts_path, dtype=np.int64))
        images = sorted(image_dir.glob('*.png'))
        n = min(len(images), len(timestamps))
        for i in range(n):
            records.append(DSECRecord(
                sequence=seq.name,
                frame_path=str(images[i]),
                timestamp_us=int(timestamps[i]),
                tracks_path=str(tracks),
                events_path=str(events),
            ))
    if not records:
        raise RuntimeError(f'No DSEC records found under {root} split={split!r}')
    return records


def _field(dtype_names, alternatives):
    for name in alternatives:
        if name in dtype_names:
            return name
    raise KeyError(f'Could not find any of {alternatives}; dtype fields={dtype_names}')


def tracks_to_target(tracks: np.ndarray, timestamp_us: int, width: int, height: int):
    names = tracks.dtype.names or ()
    tf = _field(names, ['t', 'timestamp', 'ts'])
    xf = _field(names, ['x'])
    yf = _field(names, ['y'])
    wf = _field(names, ['w', 'width'])
    hf = _field(names, ['h', 'height'])
    cf = _field(names, ['class_id', 'class', 'label'])

    # DSEC tracks are timestamp-sorted.  Use binary search instead of scanning the
    # complete structured array for every image; fall back to a boolean mask only if
    # a non-standard unsorted track file is encountered.
    tvals = tracks[tf]
    ts = int(timestamp_us)
    i0 = int(np.searchsorted(tvals, ts, side='left'))
    i1 = int(np.searchsorted(tvals, ts, side='right'))
    selected = tracks[i0:i1]
    if selected.size and not np.all(selected[tf].astype(np.int64) == ts):
        selected = tracks[tracks[tf].astype(np.int64) == ts]
    if selected.size == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
        }

    x1 = selected[xf].astype(np.float32)
    y1 = selected[yf].astype(np.float32)
    x2 = x1 + selected[wf].astype(np.float32)
    y2 = y1 + selected[hf].astype(np.float32)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    labels = selected[cf].astype(np.int64)
    valid = (
        (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) &
        (labels >= 0) & (labels < len(DSEC_CLASSES))
    )
    return {
        'boxes': torch.from_numpy(np.ascontiguousarray(boxes[valid], dtype=np.float32)),
        'labels': torch.from_numpy(np.ascontiguousarray(labels[valid], dtype=np.int64)),
    }


def _h5_lower_bound(ds, value: int) -> int:
    """Binary-search a monotonic 1-D HDF5 dataset without loading it all into RAM."""
    lo, hi = 0, len(ds)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(ds[mid]) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def load_event_window(
    events_path: str | Path,
    anchor_timestamp_us: int,
    window_us: int = 50_000,
    direction: Literal['past', 'future'] = 'past',
    height: int = 480,
    width: int = 640,
    binary: bool = True,
) -> torch.Tensor:
    """Build a 2-channel polarity event frame for one causal/future time window."""
    if direction == 'past':
        start_abs, end_abs = anchor_timestamp_us - window_us, anchor_timestamp_us
    elif direction == 'future':
        start_abs, end_abs = anchor_timestamp_us, anchor_timestamp_us + window_us
    else:
        raise ValueError(direction)

    _ensure_hdf5_filters()
    with h5py.File(events_path, 'r') as f:
        g = f['events']
        t_ds = g['t']
        t_offset = int(f['t_offset'][()]) if 't_offset' in f else 0
        start_rel = int(start_abs - t_offset)
        end_rel = int(end_abs - t_offset)
        i0 = _h5_lower_bound(t_ds, start_rel)
        i1 = _h5_lower_bound(t_ds, end_rel)
        if i1 <= i0:
            return torch.zeros((2, height, width), dtype=torch.float32)
        x = np.asarray(g['x'][i0:i1], dtype=np.int64)
        y = np.asarray(g['y'][i0:i1], dtype=np.int64)
        p = np.asarray(g['p'][i0:i1], dtype=np.int64)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]
    # DSEC polarity may be bool or {-1,+1}; normalize to channel 0/1.
    p = (p > 0).astype(np.int64)
    arr = np.zeros((2, height, width), dtype=np.float32)
    if binary:
        arr[p, y, x] = 1.0
    else:
        np.add.at(arr, (p, y, x), 1.0)
    return torch.from_numpy(arr)


def load_rgb_frame(path: str | Path, height=480, width=640) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class DSECAlignedDataset(Dataset):
    """Timestamp-aligned DSEC frame/event samples for training or evaluation.

    ``domain='event'`` returns a two-polarity event frame. ``domain='frame'`` returns
    the RGB frame at the same annotation timestamp. For real-time causal experiments,
    use ``event_direction='past'``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None,
        domain: Literal['event', 'frame'],
        window_us: int = 50_000,
        event_direction: Literal['past', 'future'] = 'past',
        height: int = 480,
        width: int = 640,
        sequence_names: tuple[str, ...] | list[str] | None = None,
    ):
        self.records = build_dsec_manifest(root, split, sequence_names=sequence_names)
        self.domain = domain
        self.window_us = int(window_us)
        self.event_direction = event_direction
        self.height = int(height)
        self.width = int(width)
        self._tracks_cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.records)

    def _tracks(self, path):
        if path not in self._tracks_cache:
            self._tracks_cache[path] = np.load(path, mmap_mode='r')
        return self._tracks_cache[path]

    def __getitem__(self, idx):
        r = self.records[idx]
        if self.domain == 'event':
            image = load_event_window(
                r.events_path, r.timestamp_us, self.window_us, self.event_direction,
                self.height, self.width,
            )
        else:
            image = load_rgb_frame(r.frame_path, self.height, self.width)
        target = tracks_to_target(self._tracks(r.tracks_path), r.timestamp_us, self.width, self.height)
        target['image_id'] = torch.tensor([idx], dtype=torch.int64)
        meta = {'sequence': r.sequence, 'timestamp_us': r.timestamp_us, 'frame_path': r.frame_path}
        return image, target, meta


def detection_collate(batch):
    images, targets, meta = zip(*batch)
    return list(images), list(targets), list(meta)
