# Event-SNN vs RGB-ANN Embedded Object Detection — v1.0.9

Standalone research code for comparing 18 ST-VGG/ResCat + SSD detector configurations on DSEC-style automotive inputs.

For each width (16/32/64) and each family (ST-VGG/ResCat):

- **A — Event-SNN:** `Padding -> BN(input) -> Conv -> PLIF`
- **B — Strict RGB-ANN twin:** `Padding -> BN(input) -> Conv -> ReLU`
- **C — ANN-native RGB:** `Conv -> BN(output) -> ReLU`

A vs B preserves the backbone topology while replacing the spiking activation and event input with ReLU/RGB. B vs C is the ANN block-order ablation.

## v1.0.9 scope

This release intentionally returns to the **v1.0.5 training and benchmark policy**:

- fixed limited-fragment Event-SNN training; no activity rescue, adaptive thresholding, or forced spike-rate target;
- dead/near-dead/saturated PLIF layers are **reported, not corrected**;
- one unified 18-model benchmark with the six Event-SNN rows loaded from trained checkpoints;
- robust repeated NVML board-energy measurement;
- Chapter-6 analytical energy decomposition;
- **SANA-FE is the only external neuromorphic-hardware simulator implemented in this release.**

NeuroBench, SpikeSim, Sinabs/Samna/Speck preflight, and ANN ONNX hardware-export paths were removed because they are not part of this study.

## Architecture provenance

The clean implementation follows Cordone's 2022 thesis design: 4x4/stride-4 patchify stem, ST-VGG C/2C/4C scaling, ResCat concatenation stages, and the reduced SSD feature pyramid. Core Event-SNN parameter counts are regression-tested against thesis Table 5.2.

## Install

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

For SANA-FE:

```bash
pip install "sanafe>=2.2.9,<3"
```

The Colab notebook installs this explicitly before the SANA-FE section.

## Validate

```bash
python scripts/validate_release.py
pytest -q
```

## Limited-fragment Event-SNN training

This remains the v1.0.5 workflow. It is used to obtain learned rather than random spike activity; it is not presented as a full DSEC generalization benchmark.

```bash
python scripts/train_fragment_snn_matrix.py \
  --sequence-dir /content/dagr/data/DSEC_fragment/test/zurich_city_13_b \
  --epochs 5 \
  --outdir fragment_checkpoints
```

Training is ordinary fixed-budget training. The code does not force activity toward Cordone's GEN1 values and does not modify dead layers after training.

## Unified 18-model benchmark

```bash
python scripts/benchmark.py \
  --sequence-dir /content/dagr/data/DSEC_fragment/test/zurich_city_13_b \
  --windows 0 \
  --checkpoint-dir fragment_checkpoints \
  --require-trained-snn \
  --gpu-energy \
  --outdir results
```

`--windows 0` uses every available causal 50-ms window. The main output is:

```text
results/benchmark_18_models.csv
```

The benchmark also writes:

- `layer_energy/*.csv`
- `spike_activity/*.csv`
- `spike_activity_summary/*.csv`
- `snn_spike_health_summary.csv`
- `gpu_energy/*_repeats.csv`
- architecture/conv-MAC reports

### NVML protocol

The non-fast default uses:

- 3 GPU warm-up traversals;
- 5 independent active repeats;
- 10 s active interval per repeat;
- 2 s pre-idle and 2 s post-idle sampling;
- continuous NVML polling every 20 ms;
- baseline = mean of the pre/post idle medians;
- time-integrated active energy minus the baseline integral;
- mean, sample SD, and Student-t 95% CI of mJ/window;
- active-at-or-below-idle repeats marked invalid instead of clipped to fake zero.

## SANA-FE neuromorphic projection

After the unified benchmark, run:

```bash
python scripts/sanafe_benchmark.py \
  --benchmark-dir results \
  --timesteps 50 \
  --timing-model detailed
```

This requires `sanafe>=2.2.9,<3` and first runs SANA-FE's bundled smoke example.

### What is actually simulated

A naive exact expansion of these 640x480 convolutional SNNs would instantiate millions of neuron states and an impractical number of explicit convolution connections in a normal Colab session. Therefore v1.0.9 implements a clearly labeled **SANA-FE layer-aggregated Loihi-like projection**:

1. The real trained PyTorch benchmark supplies measured per-layer spike activity.
2. The Chapter-6 instrumentation supplies exact per-layer neuron counts and spike-triggered synaptic-event counts for the observed workload.
3. The projection uses Loihi-1 event-energy/latency constants from SANA-FE's bundled `arch/loihi.yaml`.
4. ST-VGG ordering and ResCat branch/join dependencies are retained in a compact SANA-FE proxy graph.
5. One proxy SANA-FE event represents the measured whole-layer work; energy is scaled to the real layer and latency to an average 1024-neuron-core workload.
6. The report separately estimates the number of 1024-neuron cores and 128-core Loihi-1 chip equivalents required by the spiking feature extractor.
7. Dense SSD classification/regression prediction heads are **not** included in SANA-FE energy. Their Chapter-6 analytical cost is reported separately and is not added to SANA-FE energy.

This is a hardware projection driven by measured activity, **not** a claim that the EFOD PLIF/BN detector has been functionally converted to Loihi or that the number is a physical Loihi measurement.

Outputs:

```text
results/sanafe_projection/sanafe_projection_results.csv
results/sanafe_projection/plans/<variant>_plan.json
results/sanafe_projection/architectures/<variant>_projection.yaml
results/sanafe_projection/raw/<variant>_result.json
```

By default the SANA-FE projection fields are also merged into the Event-SNN rows of `results/benchmark_18_models.csv`, keeping one final comparison table.

## Energy evidence: keep the domains separate

1. **Architecture:** parameters, model/state memory, dense MACs/FLOP proxy.
2. **Measured activity:** per-layer spikes, sparsity, effective Event-SNN MAC/ACC counts.
3. **Chapter-6 analytical estimate:** explicit 45-nm arithmetic/memory/addressing model.
4. **NVIDIA implementation:** measured CUDA latency/VRAM and NVML board energy.
5. **SANA-FE projection:** mapped/proxy neuromorphic performance/energy estimate under its stated Loihi-like assumptions.

Do not relabel one category as another and do not sum Chapter-6 and SANA-FE energy.

See `HARDWARE_EVALUATION.md`, `RESEARCH_PROTOCOL.md`, and `SOURCE_NOTES.md`.
