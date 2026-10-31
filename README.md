# Comparison of Frame-Based and Event-Based Computer Vision Models on Real-Time Embedded Systems

**Bachelor's Thesis | Computer Engineering | Amirkabir University of Technology (Tehran Polytechnic) | 2026**

**Author:** Seyed Sina Negahban  
**Supervisor:** Dr. Hamed Farbeh

This repository contains the thesis, research code, and Google Colab notebook for a comparative study of **frame-based artificial neural networks (ANNs)** and **event-based spiking neural networks (SNNs)** for object detection under the computational and energy constraints of real-time embedded systems.

**Research question:** Does the theoretical efficiency of event-driven, sparse computation lead to lower memory use, latency, and energy consumption when comparable vision models are executed on practical hardware?

**[Read the full thesis (PDF)](final_thesis.pdf)** · **[Explore the Colab notebook](Event_SNN_RGB_ANN_18_Model_Study.ipynb)** · **[View the implementation](event_frame_embedded_od/)**

## Overview

Conventional frame-based vision models process complete images at regular intervals, even when most of the scene is unchanged. Event cameras instead report changes in brightness, allowing spiking neural networks to potentially avoid unnecessary computation. However, fewer effective operations do not necessarily mean faster or more energy-efficient execution on hardware optimized for dense numerical workloads.

To investigate this gap, the thesis compares **18 object-detection configurations** built from two backbone families, three model widths, and three event/frame processing pipelines. The study uses aligned event and RGB data from the **DSEC** automotive dataset and evaluates model complexity, observed spike activity, measured GPU performance, analytical energy consumption, and neuromorphic hardware projections.

## Model comparison

| Dimension | Configurations |
| --- | --- |
| Backbone families | ST-VGG and ResCat |
| Backbone widths | 16, 32, and 64 channels (base width) |
| Detection architecture | SSD-style object detector |
| **A — Event-SNN** | Event input; `Padding → BN → Conv → PLIF` |
| **B — Strict RGB-ANN twin** | RGB input; `Padding → BN → Conv → ReLU` |
| **C — ANN-native RGB** | RGB input; `Conv → BN → ReLU` |

Each family and width has an A/B/C counterpart, resulting in **2 × 3 × 3 = 18 models**. The strict ANN twin is designed to preserve the SNN's backbone structure as closely as possible, while the ANN-native version provides a conventional design reference. Because the sensing modality differs, the A-versus-B comparison evaluates *paired event/SNN and RGB/ANN pipelines*, not an isolated change of neuron type.

## Evaluation methodology

The experiment considers complementary forms of evidence:

1. **Architecture and memory:** parameter counts, model size, membrane-state memory, and dense-equivalent operations.
2. **Event-driven computation:** measured spike rates, sparsity, and activity-dependent effective synaptic operations.
3. **GPU runtime and energy:** full-detector inference latency, GPU memory use, and NVIDIA Tesla T4 board-energy measurements with repeated NVML sampling.
4. **Analytical energy:** operation, memory-access, and addressing costs under a 45-nm reference model inspired by the methodology discussed in the thesis.
5. **Neuromorphic projection:** a SANA-FE layer-aggregated, Loihi-like estimate informed by measured SNN activity.

The evaluation uses causal **50-ms DSEC input windows**. The SANA-FE projection covers the spiking feature-extraction subgraph, **not the dense SSD prediction heads**, and is not a physical Loihi measurement.

## Key findings

- **Reduced effective computation:** The Event-SNN configurations achieved approximately **74–93% fewer activity-dependent effective operations** than their corresponding dense ANN references in the reported computational analysis.
- **No automatic GPU efficiency gain:** Despite this computational reduction, dense PyTorch/CUDA execution of the SNNs on an NVIDIA Tesla T4 was generally **slower and more energy-intensive** than the corresponding ANN pipelines.
- **Hardware matters:** Analytical energy estimates and the neuromorphic projection illustrate why sparse, event-driven computation can be advantageous when the execution architecture can exploit it directly.
- **Main conclusion:** The practical benefits of event-based vision depend on **algorithm–hardware co-design**, not on operation counts alone.

These findings concern **computational efficiency and hardware behavior**, rather than a fully trained, head-to-head object-detection accuracy benchmark.

## Repository structure

```text
.
├── README.md
├── final_thesis.pdf
├── Event_SNN_RGB_ANN_18_Model_Study.ipynb
└── event_frame_embedded_od/
    ├── README.md                 # Detailed implementation and experiment guide
    ├── requirements.txt
    ├── pyproject.toml
    ├── RESEARCH_PROTOCOL.md
    ├── HARDWARE_EVALUATION.md
    ├── SOURCE_NOTES.md
    ├── VALIDATION.md
    ├── LICENSE
    ├── src/efod/               # Data, SNN/ANN models, training, evaluation, hardware
    ├── scripts/                # Training, benchmarking, SANA-FE, validation
    └── tests/
```

## Getting started

### Option 1: Google Colab (recommended)

Open [`Event_SNN_RGB_ANN_18_Model_Study.ipynb`](Event_SNN_RGB_ANN_18_Model_Study.ipynb) in Google Colab and run the cells in order. The notebook is designed for a fresh Colab runtime: it reconstructs the standalone source tree, installs dependencies, retrieves a public DSEC example fragment, trains the limited-fragment SNN models, runs the comparative benchmark, and generates plots and exportable results. **A GPU runtime is recommended** for performance and energy measurements.

### Option 2: Run the Python package locally

Requires **Python 3.10+**. From the implementation directory:

```bash
cd event_frame_embedded_od
python -m pip install -e ".[dev]"
python scripts/validate_release.py
pytest -q
```

To prepare the example dataset, the Colab notebook uses the public DSEC fragment download workflow provided by [DAGr](https://github.com/uzh-rpg/dagr). After making the fragment available locally, run:

```bash
# Train the six Event-SNN configurations on a limited data fragment.
python scripts/train_fragment_snn_matrix.py \
  --sequence-dir /path/to/DSEC_fragment/test/zurich_city_13_b \
  --epochs 5 \
  --outdir fragment_checkpoints

# Benchmark all 18 configurations.
python scripts/benchmark.py \
  --sequence-dir /path/to/DSEC_fragment/test/zurich_city_13_b \
  --windows 0 \
  --checkpoint-dir fragment_checkpoints \
  --require-trained-snn \
  --gpu-energy \
  --outdir results
```

The main comparison is saved to `results/benchmark_18_models.csv`, alongside per-layer activity, energy, and diagnostic reports. The `--gpu-energy` measurements require a compatible NVIDIA GPU with NVML access.

**Optional SANA-FE projection:**

```bash
python -m pip install "sanafe>=2.2.9,<3"
python scripts/sanafe_benchmark.py \
  --benchmark-dir results \
  --timesteps 50 \
  --timing-model detailed
```

For complete setup details, output descriptions, and implementation-specific assumptions, see [`event_frame_embedded_od/README.md`](event_frame_embedded_od/README.md).

## Scope and limitations

- The SNNs are trained using a **limited DSEC fragment** to obtain learned spike activity; this is not a full-dataset generalization study.
- Computational characterization on the same fragment should **not** be interpreted as independent detection-accuracy validation.
- GPU latency and energy measurements describe the **specific PyTorch/CUDA implementation** and measured hardware, not all possible SNN implementations.
- SANA-FE results are **layer-aggregated neuromorphic projections**, not exact detector deployment or measured neuromorphic-chip performance.
- Analytical, GPU-measured, and SANA-FE-projected energy values have **different assumptions and scopes** and should not be combined into a single hardware measurement.

## Thesis and attribution

**Seyed Sina Negahban** (2026). *Comparison of Frame-Based and Event-Based Computer Vision Models on Real-Time Embedded Systems*. Bachelor's thesis, Department of Computer Engineering, Amirkabir University of Technology. Supervised by Dr. Hamed Farbeh.

The implementation is an independent research reconstruction informed by earlier ST-VGG/ResCat spiking-vision research, including work by **Loïc Cordone**. It is not an official implementation by those authors. See [`SOURCE_NOTES.md`](event_frame_embedded_od/SOURCE_NOTES.md) and the thesis bibliography for methodological attribution.

## License

The standalone implementation in `event_frame_embedded_od/` is provided under the **MIT License** (see [`LICENSE`](event_frame_embedded_od/LICENSE)). The thesis PDF, third-party datasets, and external simulation tools may have separate usage and licensing terms.
