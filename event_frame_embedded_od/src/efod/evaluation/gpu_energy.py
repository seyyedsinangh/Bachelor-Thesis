from __future__ import annotations

import math
import statistics
import threading
import time
from typing import Iterable

import torch


def _nvml_handle(index=0):
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:
        return None, None


def _sample_power_series(nv, handle, seconds: float, interval: float):
    """Return ``[(perf_counter_timestamp, watts), ...]`` for an idle interval."""
    samples = []
    end = time.perf_counter() + max(0.0, float(seconds))
    while time.perf_counter() < end:
        try:
            samples.append((time.perf_counter(), nv.nvmlDeviceGetPowerUsage(handle) / 1000.0))
        except Exception:
            break
        time.sleep(max(0.005, float(interval)))
    return samples


def _median_power(samples):
    vals = [float(v) for _, v in samples if math.isfinite(float(v))]
    return statistics.median(vals) if vals else float('nan')


def _mean_power(samples):
    vals = [float(v) for _, v in samples if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else float('nan')


def _integrate_power(samples, t0: float, t1: float):
    """Time-integrate sampled board power over [t0,t1] in joules.

    NVML is sampled asynchronously. The first/last observed powers are extended to the
    exact interval boundaries, and trapezoidal integration is used between observations.
    """
    if t1 <= t0:
        return float('nan')
    pts = [(float(t), float(p)) for t, p in samples if t0 <= t <= t1 and math.isfinite(float(p))]
    if not pts:
        return float('nan')
    pts.sort(key=lambda z: z[0])
    if len(pts) == 1:
        return pts[0][1] * (t1 - t0)
    pts = [(t0, pts[0][1]), *pts, (t1, pts[-1][1])]
    energy = 0.0
    for (ta, pa), (tb, pb) in zip(pts[:-1], pts[1:]):
        if tb > ta:
            energy += 0.5 * (pa + pb) * (tb - ta)
    return energy


# 97.5th percentile of Student t, df=1..30. For larger df use 1.96.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _summary_ci(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    nan = float('nan')
    if not vals:
        return {'n': 0, 'mean': nan, 'std': nan, 'ci95_low': nan, 'ci95_high': nan, 'ci95_halfwidth': nan}
    mean = statistics.fmean(vals)
    if n == 1:
        return {'n': 1, 'mean': mean, 'std': nan, 'ci95_low': nan, 'ci95_high': nan, 'ci95_halfwidth': nan}
    std = statistics.stdev(vals)
    crit = _T975.get(n - 1, 1.96)
    half = crit * std / math.sqrt(n)
    return {'n': n, 'mean': mean, 'std': std, 'ci95_low': mean - half, 'ci95_high': mean + half, 'ci95_halfwidth': half}


def _reset_for_sequence(model):
    if hasattr(model, 'reset_state'):
        model.reset_state()


def _run_sequence_until(model, inputs, duration_s: float, state_mode: str):
    """Run repeated chronological traversals and return (windows, elapsed_s)."""
    device = inputs[0].device
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + float(duration_s)
    with torch.no_grad():
        while time.perf_counter() < deadline:
            # Each repeated traversal is a new copy of the finite workload. Preserve SNN
            # state *inside* a traversal, but do not connect the final window back to first.
            _reset_for_sequence(model)
            for x in inputs:
                if state_mode == 'reset':
                    _reset_for_sequence(model)
                model.forward_raw(x)
                n += 1
                if time.perf_counter() >= deadline:
                    break
    torch.cuda.synchronize(device)
    return n, time.perf_counter() - t0


def measure_gpu_energy_sequence(
    model,
    inputs,
    duration_s: float = 10.0,
    idle_s: float = 2.0,
    poll_interval: float = 0.02,
    state_mode: str = 'stream',
    repeats: int = 5,
    warmup_traversals: int = 3,
    settle_s: float = 0.5,
):
    """Robust NVML board-energy benchmark over a real input sequence.

    Protocol for each model:
      1. Warm the model/CUDA kernels with several full sequence traversals.
      2. Repeat the measurement ``repeats`` times (default 5).
      3. Before each active run, sample idle board power and use its median.
      4. Execute for >= ``duration_s`` (default 10 s) while continuously polling NVML.
      5. Sample post-run idle power; baseline = mean(pre-median, post-median).
      6. Integrate active board power over time and subtract baseline*elapsed.
      7. If active energy is not above the baseline, mark dynamic energy invalid (NaN),
         rather than clipping it to a fake 0 mJ.
      8. Report mean, sample SD, and Student-t 95% CI across repeated runs.

    Values are NVIDIA *board* energy for this PyTorch implementation, not a
    neuromorphic-hardware estimate.
    """
    nan = float('nan')
    if state_mode not in ('stream', 'reset'):
        raise ValueError(state_mode)
    if not inputs or inputs[0].device.type != 'cuda':
        return {
            'gpu_seq_energy_status': 'unavailable_no_cuda',
            'gpu_seq_energy_dynamic_mj': nan,
            'gpu_seq_energy_total_mj': nan,
            'gpu_seq_repeat_rows': [],
        }

    device = inputs[0].device
    nv, handle = _nvml_handle(device.index or 0)
    if handle is None:
        return {
            'gpu_seq_energy_status': 'unavailable_nvml',
            'gpu_seq_energy_dynamic_mj': nan,
            'gpu_seq_energy_total_mj': nan,
            'gpu_seq_repeat_rows': [],
        }

    model.eval()
    # Warm kernels, allocations, and GPU clocks before the first baseline.
    with torch.no_grad():
        for _ in range(max(1, int(warmup_traversals))):
            _reset_for_sequence(model)
            for x in inputs:
                if state_mode == 'reset':
                    _reset_for_sequence(model)
                model.forward_raw(x)
    torch.cuda.synchronize(device)
    if settle_s > 0:
        time.sleep(settle_s)

    repeat_rows = []
    for rep in range(max(1, int(repeats))):
        # PRE idle baseline.
        pre = _sample_power_series(nv, handle, idle_s, poll_interval)
        pre_med = _median_power(pre)

        active_samples = []
        stop = threading.Event()

        def poll_active():
            while not stop.is_set():
                try:
                    active_samples.append((time.perf_counter(), nv.nvmlDeviceGetPowerUsage(handle) / 1000.0))
                except Exception:
                    break
                time.sleep(max(0.005, float(poll_interval)))

        th = threading.Thread(target=poll_active, daemon=True)
        th.start()
        # Give the sampler a chance to take a first observation before workload starts.
        time.sleep(min(0.02, max(0.005, poll_interval)))
        t0 = time.perf_counter()
        n, _ = _run_sequence_until(model, inputs, duration_s, state_mode)
        t1 = time.perf_counter()
        stop.set()
        th.join(timeout=2.0)
        elapsed = t1 - t0

        # POST idle baseline. Give kernels/power state a short settling interval first.
        if settle_s > 0:
            time.sleep(settle_s)
        post = _sample_power_series(nv, handle, idle_s, poll_interval)
        post_med = _median_power(post)
        baseline = (pre_med + post_med) / 2.0 if math.isfinite(pre_med) and math.isfinite(post_med) else nan

        total_j = _integrate_power(active_samples, t0, t1)
        active_mean = total_j / elapsed if math.isfinite(total_j) and elapsed > 0 else _mean_power(active_samples)
        sec_per_window = elapsed / max(n, 1)
        total_mj_per_window = total_j / max(n, 1) * 1000.0 if math.isfinite(total_j) else nan
        dynamic_j = total_j - baseline * elapsed if math.isfinite(total_j) and math.isfinite(baseline) else nan
        valid_dynamic = math.isfinite(dynamic_j) and dynamic_j > 0 and n > 0
        dynamic_mj_per_window = dynamic_j / n * 1000.0 if valid_dynamic else nan
        dynamic_w = dynamic_j / elapsed if valid_dynamic and elapsed > 0 else nan
        edp = dynamic_mj_per_window * sec_per_window * 1000.0 if valid_dynamic else nan

        repeat_rows.append({
            'repeat': rep + 1,
            'status': 'ok' if valid_dynamic else 'invalid_active_not_above_idle',
            'windows': n,
            'elapsed_s': elapsed,
            'sec_per_window': sec_per_window,
            'idle_pre_median_w': pre_med,
            'idle_post_median_w': post_med,
            'idle_baseline_w': baseline,
            'active_mean_w': active_mean,
            'dynamic_mean_w': dynamic_w,
            'active_power_samples': len(active_samples),
            'total_energy_mj_per_window': total_mj_per_window,
            'dynamic_energy_mj_per_window': dynamic_mj_per_window,
            'dynamic_edp_mj_ms': edp,
        })

    total_s = _summary_ci([r['total_energy_mj_per_window'] for r in repeat_rows])
    dyn_s = _summary_ci([r['dynamic_energy_mj_per_window'] for r in repeat_rows])
    edp_s = _summary_ci([r['dynamic_edp_mj_ms'] for r in repeat_rows])
    baseline_s = _summary_ci([r['idle_baseline_w'] for r in repeat_rows])
    active_s = _summary_ci([r['active_mean_w'] for r in repeat_rows])
    dyn_power_s = _summary_ci([r['dynamic_mean_w'] for r in repeat_rows])
    pre_s = _summary_ci([r['idle_pre_median_w'] for r in repeat_rows])
    post_s = _summary_ci([r['idle_post_median_w'] for r in repeat_rows])

    status = 'ok' if dyn_s['n'] == len(repeat_rows) else ('partial' if dyn_s['n'] > 0 else 'invalid_no_positive_dynamic_runs')
    return {
        # Backward-compatible primary fields now mean repeated-run averages.
        'gpu_seq_energy_status': status,
        'gpu_seq_power_idle_w': baseline_s['mean'],
        'gpu_seq_power_idle_pre_w': pre_s['mean'],
        'gpu_seq_power_idle_post_w': post_s['mean'],
        'gpu_seq_power_active_w': active_s['mean'],
        'gpu_seq_power_dynamic_w': dyn_power_s['mean'],
        'gpu_seq_energy_total_mj': total_s['mean'],
        'gpu_seq_energy_dynamic_mj': dyn_s['mean'],
        'gpu_seq_edp_dynamic_mj_ms': edp_s['mean'],
        # Uncertainty / quality fields requested for the final benchmark.
        'gpu_seq_energy_total_std_mj': total_s['std'],
        'gpu_seq_energy_total_ci95_low_mj': total_s['ci95_low'],
        'gpu_seq_energy_total_ci95_high_mj': total_s['ci95_high'],
        'gpu_seq_energy_total_ci95_halfwidth_mj': total_s['ci95_halfwidth'],
        'gpu_seq_energy_dynamic_std_mj': dyn_s['std'],
        'gpu_seq_energy_dynamic_ci95_low_mj': dyn_s['ci95_low'],
        'gpu_seq_energy_dynamic_ci95_high_mj': dyn_s['ci95_high'],
        'gpu_seq_energy_dynamic_ci95_halfwidth_mj': dyn_s['ci95_halfwidth'],
        'gpu_seq_edp_dynamic_std_mj_ms': edp_s['std'],
        'gpu_seq_energy_repeats_requested': len(repeat_rows),
        'gpu_seq_energy_repeats_valid': dyn_s['n'],
        'gpu_seq_windows': sum(r['windows'] for r in repeat_rows),
        'gpu_seq_windows_per_repeat_mean': statistics.fmean([r['windows'] for r in repeat_rows]),
        'gpu_seq_elapsed_s': sum(r['elapsed_s'] for r in repeat_rows),
        'gpu_seq_active_duration_target_s': float(duration_s),
        'gpu_seq_idle_duration_s': float(idle_s),
        'gpu_seq_poll_interval_s': float(poll_interval),
        'gpu_seq_state_mode': state_mode,
        'gpu_seq_repeat_rows': repeat_rows,
    }


def measure_gpu_energy(model, input_tensor, duration_s=10.0, idle_s=2.0, poll_interval=.02, state_mode='stream', repeats=5):
    """Single-input convenience wrapper around the repeated sequence protocol."""
    result = measure_gpu_energy_sequence(
        model,
        [input_tensor],
        duration_s=duration_s,
        idle_s=idle_s,
        poll_interval=poll_interval,
        state_mode=state_mode,
        repeats=repeats,
    )
    # Preserve legacy key names for external callers.
    mapping = {
        'gpu_seq_power_idle_w': 'gpu_power_idle_w',
        'gpu_seq_power_active_w': 'gpu_power_active_w',
        'gpu_seq_power_dynamic_w': 'gpu_power_dynamic_w',
        'gpu_seq_energy_total_mj': 'gpu_energy_total_mj',
        'gpu_seq_energy_dynamic_mj': 'gpu_energy_dynamic_mj',
        'gpu_seq_edp_dynamic_mj_ms': 'gpu_edp_dynamic_mj_ms',
    }
    for old, new in mapping.items():
        result[new] = result.get(old, float('nan'))
    return result
