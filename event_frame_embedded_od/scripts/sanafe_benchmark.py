#!/usr/bin/env python
from __future__ import annotations

"""Run the SANA-FE projection for the six trained Event-SNN variants.

This consumes outputs already produced by ``scripts/benchmark.py``:

* benchmark_18_models.csv
* spike_activity_summary/<variant>_event_snn.csv
* layer_energy/<variant>_event_snn.csv

The detector is too large to naively instantiate every 640x480 neuron and every
convolution edge in SANA-FE in a normal Colab runtime.  We therefore use the
explicitly documented layer-aggregated projection implemented in
``efod.hardware.sanafe_adapter``.  It preserves measured trained spike activity,
layer neuron counts, spike-triggered synaptic-event counts, ST-VGG/ResCat graph
ordering, and Loihi-1 event costs from SANA-FE's reference architecture.

Dense SSD prediction heads are deliberately excluded from the SANA-FE result;
their Chapter-6 analytical cost remains reported separately.  Never add those
two energy models together and call the sum a SANA-FE measurement.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from efod import MODEL_SPECS, build_model
from efod.hardware import (
    make_projection_plan,
    run_sanafe_projection,
    sanafe_smoke_test,
    save_plan,
    write_proxy_architecture_yaml,
)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _merge_results(main_csv: Path, sana_df: pd.DataFrame) -> None:
    df = pd.read_csv(main_csv)
    keyed = sana_df.set_index("variant")
    new_cols = [c for c in sana_df.columns if c != "variant"]
    for col in new_cols:
        if col not in df.columns:
            df[col] = pd.NA
    mask = df["pipeline"] == "event_snn"
    for idx in df.index[mask]:
        variant = df.at[idx, "variant"]
        if variant not in keyed.index:
            continue
        for col in new_cols:
            df.at[idx, col] = keyed.at[variant, col]
    df.to_csv(main_csv, index=False)
    main_md = main_csv.with_suffix(".md")
    main_md.write_text(df.to_markdown(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="SANA-FE projection of the six trained Event-SNNs")
    p.add_argument("--benchmark-dir", default="results", help="Directory produced by scripts/benchmark.py")
    p.add_argument("--outdir", default=None, help="Default: <benchmark-dir>/sanafe_projection")
    p.add_argument("--timesteps", type=int, default=50, help="Proxy streaming timesteps simulated in SANA-FE")
    p.add_argument("--timing-model", choices=["simple", "detailed", "cycle"], default="detailed")
    p.add_argument("--allow-random", action="store_true", help="Allow a random Event-SNN benchmark (not recommended)")
    p.add_argument("--no-merge", action="store_true", help="Do not append SANA-FE columns to benchmark_18_models.csv")
    a = p.parse_args()

    bench_dir = Path(a.benchmark_dir)
    main_csv = _require(bench_dir / "benchmark_18_models.csv")
    out = Path(a.outdir) if a.outdir else bench_dir / "sanafe_projection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plans").mkdir(exist_ok=True)
    (out / "architectures").mkdir(exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)

    smoke = sanafe_smoke_test()
    (out / "sanafe_smoke.json").write_text(json.dumps(smoke, indent=2, default=str))
    if not smoke.get("available", False):
        raise RuntimeError(
            "SANA-FE is not available. Install it with `pip install 'sanafe>=2.2.9,<3'`. "
            f"Smoke error: {smoke.get('error', 'unknown')}"
        )

    bench = pd.read_csv(main_csv)
    snn = bench[bench.pipeline == "event_snn"].copy()
    if len(snn) != 6:
        raise ValueError(f"Expected six event_snn rows, found {len(snn)}")
    if not a.allow_random and not snn.weights_state.astype(str).str.startswith("trained").all():
        bad = snn.loc[~snn.weights_state.astype(str).str.startswith("trained"), ["variant", "weights_state"]]
        raise ValueError("SANA-FE requires the trained-SNN benchmark by default:\n" + bad.to_string(index=False))

    rows = []
    for _, br in snn.iterrows():
        variant = str(br.variant)
        print(f"\n=== SANA-FE layer-aggregated Loihi-1-like projection | {variant} ===", flush=True)
        if variant not in MODEL_SPECS:
            raise KeyError(variant)
        spike_csv = _require(bench_dir / "spike_activity_summary" / f"{variant}_event_snn.csv")
        energy_csv = _require(bench_dir / "layer_energy" / f"{variant}_event_snn.csv")
        spike_summary = pd.read_csv(spike_csv)
        layer_energy = pd.read_csv(energy_csv)
        windows = int(br.get("benchmark_windows", 0))
        if windows <= 0:
            raise ValueError(f"Invalid benchmark_windows for {variant}: {windows}")

        # The learned weights are not needed here because the measured activity
        # and event counts have already been captured by the PyTorch benchmark.
        model = build_model(variant, "event_snn").eval()
        plan = make_projection_plan(model, spike_summary, layer_energy, windows=windows)
        plan_path = save_plan(plan, out / "plans" / f"{variant}_plan.json")
        arch_path = write_proxy_architecture_yaml(plan, out / "architectures" / f"{variant}_projection.yaml")
        result = run_sanafe_projection(plan, arch_path, timesteps=a.timesteps, timing_model=a.timing_model)
        result.update({
            "variant": variant,
            "model": br.get("model", MODEL_SPECS[variant]["display"]),
            "weights_state": br.get("weights_state", ""),
            "benchmark_windows": windows,
            "seq_spike_rate": br.get("seq_spike_rate", float("nan")),
            "seq_spike_dead_layers": br.get("seq_spike_dead_layers", float("nan")),
            "sanafe_projection_plan_json": str(plan_path),
            "sanafe_projection_architecture_yaml": str(arch_path),
        })
        (out / "raw" / f"{variant}_result.json").write_text(json.dumps(result, indent=2, default=str))
        rows.append(result)
        print({k: result.get(k) for k in (
            "sanafe_projected_spiking_subgraph_energy_mJ_per_window",
            "sanafe_projected_spiking_subgraph_latency_ms_per_window",
            "sanafe_projected_spiking_subgraph_graph_critical_path_ms_per_window",
            "sanafe_naive_loihi1_core_capacity_estimate",
            "sanafe_naive_loihi1_chip_capacity_equivalents",
        )})

    sdf = pd.DataFrame(rows)
    summary_csv = out / "sanafe_projection_results.csv"
    sdf.to_csv(summary_csv, index=False)
    (out / "sanafe_projection_results.md").write_text(sdf.to_markdown(index=False))

    if not a.no_merge:
        _merge_results(main_csv, sdf)
        print("Merged SANA-FE columns into", main_csv)
    print("Saved", summary_csv)


if __name__ == "__main__":
    main()
