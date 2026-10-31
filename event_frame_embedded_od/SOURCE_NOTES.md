# Source notes — v1.0.9

## Cordone architecture and training provenance

The ST-VGG/ResCat design, patchify stem, no-reset training concept, and Chapter-6 energy methodology are reconstructed from Loïc Cordone's 2022 thesis and associated public code/paper. The SSD/DSEC implementation in this package is standalone.

## Chapter-6 analytical energy

The analytical estimator mirrors the thesis decomposition into memory accesses, synaptic operations, and addressing. The default 45-nm reference uses 0.1 pJ/add, 3.1 pJ/multiply, and SRAM-access interpolation from the thesis. It is an analytical model, not hardware measurement.

## SANA-FE

SANA-FE is the only external neuromorphic simulator implemented in v1.0.9. The current public SANA-FE API supports mapped SNN construction, `SpikingChip`, timestep simulation, energy/timing results, and a bundled Loihi-1 architecture.

The projection constants in `efod.hardware.sanafe_adapter` follow SANA-FE's bundled `arch/loihi.yaml` (Davies et al. Loihi-1 values): 1024 neurons/core, 128 cores/chip, soma event costs, convolution-synapse cost, axon costs, and directional mesh-hop costs.

Because the detector is too large for convenient exact expansion in Colab, EFOD uses a layer-aggregated proxy whose workload is derived from measured trained-SNN layer activity and Chapter-6 event counters. This is explicitly labeled a projection rather than exact mapping or physical measurement.

## Removed external paths

NeuroBench, SpikeSim, Sinabs/Samna/Speck, and ANN ONNX hardware-export helpers present in earlier development versions are intentionally absent from v1.0.9.
