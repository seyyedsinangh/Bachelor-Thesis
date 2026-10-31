from __future__ import annotations

"""SANA-FE projection utilities for the Event-SNN feature extractor.

The full 640x480 ST-VGG/ResCat detector contains millions of spiking state
variables and, if expanded naively, billions of convolution synapses.  SANA-FE's
Python network representation stores neurons and edges explicitly, so an exact
full expansion is not practical in a normal Colab runtime.

This module therefore implements a *layer-aggregated, core-parallel projection*:

* measured per-layer spike activity comes from the real PyTorch/DSEC benchmark;
* exact per-layer neuron counts and spike-triggered synaptic-event counts come
  from the benchmark's Chapter-6 instrumentation;
* Loihi-1 event energy/latency constants are taken from SANA-FE's bundled
  ``arch/loihi.yaml``;
* each real spiking block is represented by one proxy neuron in SANA-FE;
* proxy hardware-unit energy is scaled to the whole real layer, while latency is
  scaled to the average 1024-neuron Loihi core workload (parallel-core model);
* ResCat branch/join topology is retained at the proxy-graph level;
* dense SSD classification/regression heads are intentionally excluded.

The output must be described as a "SANA-FE layer-aggregated Loihi-like
projection", not as a measurement of a physical Loihi chip or an exact mapping.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import importlib.metadata
import json
import math
import re

import pandas as pd


# SANA-FE arch/loihi.yaml (Loihi-1 values; Davies et al. 2018 source in file).
LOIHI1 = {
    "max_neurons_per_core": 1024,
    "cores_per_chip": 128,  # 32 tiles x 4 cores in bundled SANA-FE architecture
    "soma_access_energy_j": 51.2e-12,
    "soma_update_energy_j": 21.6e-12,
    "soma_spike_energy_j": 69.3e-12,
    "soma_access_latency_s": 6.0e-9,
    "soma_update_latency_s": 3.7e-9,
    "soma_spike_latency_s": 30.0e-9,
    "conv_synapse_energy_j": 24.0e-12,
    "conv_synapse_latency_s": 3.1e-9,
    "axon_out_energy_j": 111.0e-12,
    "axon_out_latency_s": 5.1e-9,
    "axon_in_energy_j": 0.0,
    "axon_in_latency_s": 16.0e-9,
    # Mean of N/E/S/W values in SANA-FE's bundled loihi.yaml.
    "mean_hop_energy_j": 3.6e-12,
    "mean_hop_latency_s": 5.3e-9,
}


@dataclass
class ProjectedLayer:
    name: str
    neurons: int
    output_spikes: float
    spike_rate: float
    input_spikes: float
    synaptic_events: float
    physical_cores: int
    avg_neurons_per_core: float
    avg_spikes_per_core: float
    packet_fanout_per_input_spike: int
    network_packets: float
    soma_energy_j: float
    synapse_energy_j: float
    network_energy_j: float
    total_energy_j: float
    soma_latency_s: float
    synapse_latency_s: float
    network_latency_s: float
    stage_latency_s: float


@dataclass
class ProjectionPlan:
    variant: str
    windows: int
    layers: list[ProjectedLayer]
    edges: list[tuple[str, str]]
    joins: list[str]
    physical_core_estimate: int
    loihi1_chip_equivalents: int
    projected_energy_j_per_window: float
    projected_critical_path_s: float
    chapter6_spiking_subgraph_uJ: float
    chapter6_dense_heads_uJ: float
    packet_model: str
    target: str = "SANA-FE layer-aggregated Loihi-1-like spiking-subgraph projection"

    def to_dict(self):
        d = asdict(self)
        d["layers"] = [asdict(x) for x in self.layers]
        return d


def sanafe_version() -> str:
    try:
        return importlib.metadata.version("sanafe")
    except Exception:
        return "unknown"


def sanafe_smoke_test() -> dict:
    try:
        import sanafe
        arch, net = sanafe.load_example()
        chip = sanafe.SpikingChip(arch)
        chip.load(net)
        result = chip.sim(2)
        return {
            "available": True,
            "version": sanafe_version(),
            "timesteps_executed": result.get("timesteps_executed"),
            "energy": result.get("energy"),
            "sim_time": result.get("sim_time"),
        }
    except Exception as exc:  # pragma: no cover - depends on optional wheel
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _block_names(model) -> list[str]:
    from ..models.blocks import PreActSpikingBlock
    return [name for name, module in model.named_modules() if isinstance(module, PreActSpikingBlock)]


def proxy_graph_for_model(model) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Return real spiking block names and a compact dependency graph.

    Graph nodes include ``__input__`` and zero-cost ResCat ``__join*`` nodes.
    Regular layer nodes use the exact ``named_modules`` block names so they align
    with ``*.act`` and ``*.conv`` benchmark CSV rows.
    """
    blocks = _block_names(model)
    if not blocks:
        return [], [], []

    # ST-VGG registers its core blocks then the three SSD extra blocks in the
    # same execution order, so a chain is exact for the spiking subgraph.
    if model.cfg.spec["family"] == "stvgg":
        edges = [("__input__", blocks[0])] + list(zip(blocks[:-1], blocks[1:]))
        return blocks, edges, []

    # ResCat: retain main/residual branches and concatenation joins explicitly.
    required = [
        "backbone.core.patchify", "backbone.core.pre",
        "backbone.core.stage1.main1", "backbone.core.stage1.main2", "backbone.core.stage1.residual",
        "backbone.core.stage2.main1", "backbone.core.stage2.main2", "backbone.core.stage2.residual",
        "backbone.core.stage3.main1", "backbone.core.stage3.main2", "backbone.core.stage3.residual",
        "backbone.extras.0", "backbone.extras.1", "backbone.extras.2",
    ]
    missing = [n for n in required if n not in blocks]
    if missing:
        raise ValueError(f"Unexpected ResCat module layout; missing {missing}")
    j1, j2, j3 = "__join_stage1__", "__join_stage2__", "__join_stage3__"
    edges = [
        ("__input__", "backbone.core.patchify"),
        ("backbone.core.patchify", "backbone.core.pre"),
        ("backbone.core.pre", "backbone.core.stage1.main1"),
        ("backbone.core.stage1.main1", "backbone.core.stage1.main2"),
        ("backbone.core.pre", "backbone.core.stage1.residual"),
        ("backbone.core.stage1.main2", j1),
        ("backbone.core.stage1.residual", j1),
        (j1, "backbone.core.stage2.main1"),
        ("backbone.core.stage2.main1", "backbone.core.stage2.main2"),
        (j1, "backbone.core.stage2.residual"),
        ("backbone.core.stage2.main2", j2),
        ("backbone.core.stage2.residual", j2),
        (j2, "backbone.core.stage3.main1"),
        ("backbone.core.stage3.main1", "backbone.core.stage3.main2"),
        (j2, "backbone.core.stage3.residual"),
        ("backbone.core.stage3.main2", j3),
        ("backbone.core.stage3.residual", j3),
        (j3, "backbone.extras.0"),
        ("backbone.extras.0", "backbone.extras.1"),
        ("backbone.extras.1", "backbone.extras.2"),
    ]
    return blocks, edges, [j1, j2, j3]


def _critical_path(layers: dict[str, ProjectedLayer], edges: Iterable[tuple[str, str]], joins: set[str]) -> float:
    nodes = {"__input__", *layers.keys(), *joins}
    parents: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        parents.setdefault(dst, []).append(src)
        parents.setdefault(src, [])
    memo: dict[str, float] = {"__input__": 0.0}

    def visit(n: str, active: set[str]) -> float:
        if n in memo:
            return memo[n]
        if n in active:
            raise ValueError("Cycle in SANA-FE proxy graph")
        active.add(n)
        base = max((visit(p, active) for p in parents.get(n, [])), default=0.0)
        own = 0.0 if n in joins else layers[n].stage_latency_s
        memo[n] = base + own
        active.remove(n)
        return memo[n]

    for n in nodes:
        visit(n, set())
    return max(memo.values(), default=0.0)


def make_projection_plan(
    model,
    spike_summary: pd.DataFrame,
    layer_energy: pd.DataFrame,
    windows: int,
) -> ProjectionPlan:
    """Build a measured-activity Loihi-like projection plan.

    ``spike_summary`` is ``spike_activity_summary/<variant>_event_snn.csv``.
    ``layer_energy`` is ``layer_energy/<variant>_event_snn.csv`` from the same
    benchmark.  The latter contains one row per convolution per DSEC window.
    """
    if windows <= 0:
        raise ValueError("windows must be > 0")
    block_names, edges, joins = proxy_graph_for_model(model)
    if not block_names:
        raise ValueError("Model has no spiking blocks")

    spike_by_name = {str(r["name"]): r for _, r in spike_summary.iterrows()}
    energy_spiking = layer_energy[layer_energy["kind"] == "SNN/spiking"].copy()
    if energy_spiking.empty:
        raise ValueError("No SNN/spiking rows found in layer-energy CSV")
    energy_grouped = energy_spiking.groupby("name", sort=False).mean(numeric_only=True)

    out: list[ProjectedLayer] = []
    for block_name in block_names:
        act_name = block_name + ".act"
        conv_name = block_name + ".conv"
        if act_name not in spike_by_name:
            raise KeyError(f"Missing activity row {act_name}")
        if conv_name not in energy_grouped.index:
            raise KeyError(f"Missing energy row {conv_name}")
        a = spike_by_name[act_name]
        e = energy_grouped.loc[conv_name]

        neurons_observed = float(a.get("neurons_observed", 0.0))
        neurons = max(1, int(round(neurons_observed / windows)))
        output_spikes = float(a.get("spikes", 0.0)) / windows
        rate = float(a.get("rate", output_spikes / neurons))
        input_spikes = max(0.0, float(e.get("theta_in", 0.0)))
        syn_acc = max(0.0, float(e.get("synaptic_accs", 0.0)))
        # Thesis instrumentation adds one per-output-neuron ACC for PLIF state
        # update. Remove it to isolate spike-triggered synapse lookups.
        synaptic_events = max(0.0, syn_acc - neurons)

        cores = max(1, math.ceil(neurons / LOIHI1["max_neurons_per_core"]))
        n_per_core = neurons / cores
        spikes_per_core = output_spikes / cores

        if input_spikes > 0.0 and synaptic_events > 0.0:
            synapses_per_input_spike = synaptic_events / input_spikes
            packet_fanout = max(1, math.ceil(synapses_per_input_spike / LOIHI1["max_neurons_per_core"]))
            packet_fanout = min(packet_fanout, cores)
            packets = input_spikes * packet_fanout
        else:
            packet_fanout = 0
            packets = 0.0

        soma_e = (
            neurons * (LOIHI1["soma_access_energy_j"] + LOIHI1["soma_update_energy_j"])
            + output_spikes * LOIHI1["soma_spike_energy_j"]
        )
        syn_e = synaptic_events * LOIHI1["conv_synapse_energy_j"]
        net_e = packets * (
            LOIHI1["axon_out_energy_j"] + LOIHI1["mean_hop_energy_j"] + LOIHI1["axon_in_energy_j"]
        )

        # Core-parallel critical-path estimate: distribute real work evenly over
        # the estimated 1024-neuron destination cores. This is a mapping
        # projection, not an exact placement/routing solution.
        soma_t = (
            n_per_core * (LOIHI1["soma_access_latency_s"] + LOIHI1["soma_update_latency_s"])
            + spikes_per_core * LOIHI1["soma_spike_latency_s"]
        )
        syn_t = (synaptic_events / cores) * LOIHI1["conv_synapse_latency_s"]
        net_t = (packets / cores) * (
            LOIHI1["axon_out_latency_s"] + LOIHI1["mean_hop_latency_s"] + LOIHI1["axon_in_latency_s"]
        )

        out.append(ProjectedLayer(
            name=block_name,
            neurons=neurons,
            output_spikes=output_spikes,
            spike_rate=rate,
            input_spikes=input_spikes,
            synaptic_events=synaptic_events,
            physical_cores=cores,
            avg_neurons_per_core=n_per_core,
            avg_spikes_per_core=spikes_per_core,
            packet_fanout_per_input_spike=packet_fanout,
            network_packets=packets,
            soma_energy_j=soma_e,
            synapse_energy_j=syn_e,
            network_energy_j=net_e,
            total_energy_j=soma_e + syn_e + net_e,
            soma_latency_s=soma_t,
            synapse_latency_s=syn_t,
            network_latency_s=net_t,
            stage_latency_s=soma_t + syn_t + net_t,
        ))

    layer_map = {x.name: x for x in out}
    physical_cores = sum(x.physical_cores for x in out)
    critical = _critical_path(layer_map, edges, set(joins))
    projected_energy = sum(x.total_energy_j for x in out)

    # Keep the Chapter-6 subgraph/head split visible, but never add it to SANA-FE
    # energy: they are different energy models.
    grouped_kind = layer_energy.groupby("kind", sort=False)["total_energy_pj"].mean() if "total_energy_pj" in layer_energy else pd.Series(dtype=float)
    # Correct averaging: layer_energy has one row/layer/window. Sum layer means.
    kind_means = layer_energy.groupby(["kind", "name"], sort=False)["total_energy_pj"].mean().reset_index()
    ch6_snn_uJ = float(kind_means.loc[kind_means.kind == "SNN/spiking", "total_energy_pj"].sum() / 1e6)
    ch6_dense_uJ = float(kind_means.loc[kind_means.kind == "ANN/dense", "total_energy_pj"].sum() / 1e6)

    return ProjectionPlan(
        variant=model.cfg.variant,
        windows=windows,
        layers=out,
        edges=edges,
        joins=joins,
        physical_core_estimate=physical_cores,
        loihi1_chip_equivalents=max(1, math.ceil(physical_cores / LOIHI1["cores_per_chip"])),
        projected_energy_j_per_window=projected_energy,
        projected_critical_path_s=critical,
        chapter6_spiking_subgraph_uJ=ch6_snn_uJ,
        chapter6_dense_heads_uJ=ch6_dense_uJ,
        packet_model=(
            "one-hop Loihi-like packet estimate; packets/input-spike = "
            "ceil(avg spike-triggered synapse fanout / 1024), capped at destination core count"
        ),
    )


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def write_proxy_architecture_yaml(plan: ProjectionPlan, path: str | Path) -> Path:
    """Write a tiny SANA-FE architecture whose unit costs encode the projection.

    One proxy core is used per graph node. Energy values are *whole-layer per
    window* costs; latency values are the projected average physical-core costs.
    NoC/axon costs are set to zero because their estimated cost is already folded
    into each destination layer's proxy synapse unit.
    """
    path = Path(path)
    layer_count = len(plan.layers)
    proxy_cores = 1 + layer_count + len(plan.joins)
    lines: list[str] = []
    ap = lines.append
    ap("architecture:")
    ap(f"  name: efod_{_safe_name(plan.variant)}_loihi_projection")
    ap("  attributes:")
    ap("    topology: mesh")
    ap("    width: 1")
    ap("    height: 1")
    ap("    link_buffer_size: 16")
    ap("    sync_model: fixed")
    ap("    latency_sync: 0.0")
    ap("  tile:")
    ap("    - name: proxy_tile[0..0]")
    ap("      attributes:")
    for d in ("north", "east", "south", "west"):
        ap(f"        energy_{d}_hop: 0.0")
        ap(f"        latency_{d}_hop: 0.0")
    ap("      core:")
    ap(f"        - name: proxy_core[0..{proxy_cores-1}]")
    ap("          attributes:")
    ap("            buffer_position: soma")
    ap("            buffer_inside_unit: false")
    ap("            max_neurons_supported: 1")
    ap("          axon_in:")
    ap("            - name: proxy_in")
    ap("              attributes:")
    ap("                energy_message_in: 0.0")
    ap("                latency_message_in: 0.0")
    ap("          soma:")
    # Source and join proxies have zero cost but use LIF to emit one control spike.
    for unit in ("proxy_source_soma", "proxy_join_soma"):
        ap(f"            - name: {unit}")
        ap("              attributes:")
        ap("                model: leaky_integrate_fire")
        ap("                energy_access_neuron: 0.0")
        ap("                latency_access_neuron: 0.0")
        ap("                energy_update_neuron: 0.0")
        ap("                latency_update_neuron: 0.0")
        ap("                energy_spike_out: 0.0")
        ap("                latency_spike_out: 0.0")
    for i, layer in enumerate(plan.layers):
        ap(f"            - name: layer_{i}_soma")
        ap("              attributes:")
        ap("                model: leaky_integrate_fire")
        # The proxy neuron updates/spikes once; scale its energy to all real
        # neurons/spikes and its latency to the average physical core workload.
        access_e = layer.neurons * LOIHI1["soma_access_energy_j"]
        update_e = layer.neurons * LOIHI1["soma_update_energy_j"]
        spike_e = layer.output_spikes * LOIHI1["soma_spike_energy_j"]
        access_t = layer.avg_neurons_per_core * LOIHI1["soma_access_latency_s"]
        update_t = layer.avg_neurons_per_core * LOIHI1["soma_update_latency_s"]
        spike_t = layer.avg_spikes_per_core * LOIHI1["soma_spike_latency_s"]
        ap(f"                energy_access_neuron: {access_e:.17g}")
        ap(f"                latency_access_neuron: {access_t:.17g}")
        ap(f"                energy_update_neuron: {update_e:.17g}")
        ap(f"                latency_update_neuron: {update_t:.17g}")
        ap(f"                energy_spike_out: {spike_e:.17g}")
        ap(f"                latency_spike_out: {spike_t:.17g}")
    ap("          dendrite:")
    ap("            - name: proxy_dendrite")
    ap("              attributes:")
    ap("                model: accumulator")
    ap("                energy_update: 0.0")
    ap("                latency_update: 0.0")
    ap("          synapse:")
    ap("            - name: proxy_join_synapse")
    ap("              attributes:")
    ap("                model: current_based")
    ap("                energy_process_spike: 0.0")
    ap("                latency_process_spike: 0.0")
    for i, layer in enumerate(plan.layers):
        ap(f"            - name: layer_{i}_synapse")
        ap("              attributes:")
        ap("                model: current_based")
        ap(f"                energy_process_spike: {(layer.synapse_energy_j + layer.network_energy_j):.17g}")
        ap(f"                latency_process_spike: {(layer.synapse_latency_s + layer.network_latency_s):.17g}")
    ap("          axon_out:")
    ap("            - name: proxy_out")
    ap("              attributes:")
    ap("                energy_message_out: 0.0")
    ap("                latency_message_out: 0.0")
    path.write_text("\n".join(lines) + "\n")
    return path


def _energy_total_j(result: dict) -> float:
    energy = result.get("energy", 0.0)
    if isinstance(energy, (int, float)):
        return float(energy)
    if isinstance(energy, dict):
        if "total" in energy:
            return float(energy["total"])
        # Defensive fallback for API variants: sum scalar leaves.
        vals = [float(v) for v in energy.values() if isinstance(v, (int, float))]
        return sum(vals)
    raise TypeError(f"Unexpected SANA-FE energy result: {type(energy)}")


def run_sanafe_projection(
    plan: ProjectionPlan,
    architecture_yaml: str | Path,
    timesteps: int = 50,
    timing_model: str = "detailed",
) -> dict:
    """Run the layer-aggregated projection in the real SANA-FE simulator."""
    if timesteps <= 1:
        raise ValueError("Use at least 2 timesteps for the proxy streaming run")
    import sanafe

    arch = sanafe.load_arch(str(architecture_yaml))
    net = sanafe.Network()
    layer_index = {layer.name: i for i, layer in enumerate(plan.layers)}

    common_spiker = {
        "bias": 2.0,
        "threshold": 1.0,
        "reset": 0.0,
        "reset_mode": "hard",
        "leak_decay": 0.0,
        "force_update": True,
    }
    groups = {}
    groups["__input__"] = net.create_neuron_group(
        "proxy_input", 1,
        model_attributes=common_spiker,
        soma_hw_name="proxy_source_soma",
        default_synapse_hw_name="proxy_join_synapse",
    )
    for j in plan.joins:
        groups[j] = net.create_neuron_group(
            _safe_name(j), 1,
            model_attributes=common_spiker,
            soma_hw_name="proxy_join_soma",
            default_synapse_hw_name="proxy_join_synapse",
        )
    for layer in plan.layers:
        idx = layer_index[layer.name]
        groups[layer.name] = net.create_neuron_group(
            f"layer_{idx}_{_safe_name(layer.name)}", 1,
            model_attributes=common_spiker,
            soma_hw_name=f"layer_{idx}_soma",
            default_synapse_hw_name=f"layer_{idx}_synapse",
        )

    # Incoming edge synapse implementation is selected by the destination group.
    for src, dst in plan.edges:
        weight = 0.0 if dst not in plan.joins else 1.0
        groups[src][0].connect_to_neuron(groups[dst][0], {"weight": weight})

    # One proxy neuron/core. This is the compact simulation representation, not
    # the physical core count of the full detector (reported separately).
    ordered = [groups["__input__"]] + [groups[x.name] for x in plan.layers] + [groups[j] for j in plan.joins]
    cores = arch.tiles[0].cores
    if len(cores) < len(ordered):
        raise RuntimeError(f"Proxy architecture has {len(cores)} cores but needs {len(ordered)}")
    for i, g in enumerate(ordered):
        g[0].map_to_core(cores[i])

    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    result = chip.sim(timesteps, timing_model=timing_model)
    total_energy = _energy_total_j(result)
    sim_time = float(result.get("sim_time", float("nan")))
    energy_per_step = total_energy / timesteps
    time_per_step = sim_time / timesteps if sim_time == sim_time else float("nan")
    expected = plan.projected_energy_j_per_window
    rel_err = abs(energy_per_step - expected) / expected if expected > 0 else 0.0
    return {
        "sanafe_available": True,
        "sanafe_version": sanafe_version(),
        "sanafe_projection_method": "layer_aggregated_loihi1_like_proxy",
        "sanafe_projection_target": plan.target,
        "sanafe_projection_timing_model": timing_model,
        "sanafe_projection_timesteps": timesteps,
        "sanafe_proxy_node_count": 1 + len(plan.layers) + len(plan.joins),
        # Capacity-only estimates: these are NOT an actual SANA-FE placement result.
        "sanafe_naive_loihi1_core_capacity_estimate": plan.physical_core_estimate,
        "sanafe_naive_loihi1_chip_capacity_equivalents": plan.loihi1_chip_equivalents,
        # Raw proxy-simulator totals across all simulated proxy timesteps.
        "sanafe_proxy_simulated_energy_total_j": total_energy,
        # One proxy timestep represents one measured DSEC event window.
        "sanafe_projected_spiking_subgraph_energy_mJ_per_window": energy_per_step * 1e3,
        "sanafe_projected_spiking_subgraph_expected_energy_mJ_per_window": expected * 1e3,
        "sanafe_projection_internal_energy_relative_error": rel_err,
        "sanafe_projected_spiking_subgraph_latency_ms_per_window": time_per_step * 1e3,
        "sanafe_projected_spiking_subgraph_graph_critical_path_ms_per_window": plan.projected_critical_path_s * 1e3,
        "sanafe_projected_spiking_subgraph_average_power_w": (
            total_energy / sim_time if sim_time and sim_time == sim_time else float("nan")
        ),
        "sanafe_proxy_simulated_spikes": int(result.get("spikes", 0)),
        "chapter6_reference_spiking_subgraph_uJ_per_window": plan.chapter6_spiking_subgraph_uJ,
        "chapter6_reference_dense_prediction_heads_uJ_per_window": plan.chapter6_dense_heads_uJ,
        "sanafe_projection_scope": (
            "spiking backbone + spiking SSD extra feature layers only; "
            "dense SSD classification/regression prediction heads excluded"
        ),
        "sanafe_projection_activity_source": (
            "measured per-layer activity from the trained Event-SNN DSEC benchmark"
        ),
        "sanafe_projection_mapping_note": (
            "Layer-aggregated proxy simulation, not an exact neuron/synapse placement. "
            "Whole-layer energy is represented by scaled proxy hardware events; latency uses "
            "an average 1024-neuron/core parallel-workload projection. Core/chip capacity values "
            "are naive state-capacity estimates and are not the number of cores used by the compact proxy simulation."
        ),
        "sanafe_projection_packet_model": plan.packet_model,
    }


def save_plan(plan: ProjectionPlan, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(plan.to_dict(), indent=2))
    return path
