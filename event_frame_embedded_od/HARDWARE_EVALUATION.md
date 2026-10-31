# Hardware and energy evaluation plan — v1.0.9

The project now uses only three energy/performance evidence layers:

1. **Chapter-6 analytical energy** — 45-nm reference model based on operation, memory-access, and addressing counts.
2. **CUDA/NVML** — actual execution of the PyTorch implementation on the current NVIDIA GPU.
3. **SANA-FE** — the only external neuromorphic-hardware simulator used in this release.

## SANA-FE scope

SANA-FE accepts a hardware architecture plus a mapped SNN and simulates timestep-level execution, reporting energy and timing. For this detector, exact full expansion is too large for the intended Colab workflow, so the implementation is a **layer-aggregated Loihi-like projection**.

The projection consumes the measured outputs of the same trained Event-SNN benchmark:

- per-layer spike counts/rates;
- per-layer output-neuron counts;
- spike-triggered synaptic-event counts;
- graph dependencies for ST-VGG or ResCat.

Loihi-1 event constants used by the projection are the values in SANA-FE's bundled `arch/loihi.yaml`: 1024 neurons/core; 32 tiles x 4 cores; soma access/update/spike, convolution synapse, axon, and NoC-hop energy/latency values.

### Reported SANA-FE fields

- `sanafe_projected_spiking_subgraph_energy_mJ_per_window`
- `sanafe_projected_spiking_subgraph_latency_ms_per_window`
- `sanafe_projected_spiking_subgraph_graph_critical_path_ms_per_window`
- `sanafe_projected_spiking_subgraph_average_power_w`
- `sanafe_naive_loihi1_core_capacity_estimate`
- `sanafe_naive_loihi1_chip_capacity_equivalents`
- SANA-FE version, timing model, timesteps, scope, activity source, and mapping note

### Scope boundary

SANA-FE covers the **spiking backbone + spiking SSD extra feature layers**. The dense SSD classification/regression heads stay outside the SANA-FE number. Their Chapter-6 energy is exported as a separate descriptive field only.

Never report:

```text
SANA-FE energy + Chapter-6 dense-head energy
```

as a single SANA-FE measurement. They are different models.

## Why a layer-aggregated projection is used

The input is 640x480 and the feature extractor has millions of state variables. Expanding all convolution synapses explicitly would make a normal Colab simulation needlessly large. The proxy retains measured activity and graph ordering while using SANA-FE's hardware pipeline simulator on a compact representation.

Accordingly, use the wording:

> SANA-FE layer-aggregated Loihi-like projection driven by measured trained-SNN activity.

Do not call it:

> physical Loihi measurement

or

> exact functional Loihi deployment.

## NVIDIA energy protocol

The v1.0.5 robust NVML procedure is retained unchanged: 3 warm-ups, 5 x 10-s active repeats, 2-s idle sampling before and after each repeat, 20-ms polling, baseline subtraction, mean/SD/Student-t 95% CI, and invalid-run handling instead of zero clipping.
