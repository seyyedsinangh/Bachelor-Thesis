# Research protocol — v1.0.9

## Primary comparison

For each ST-VGG/ResCat width, compare:

- A Event-SNN: BN -> Conv -> PLIF on 50-ms DSEC event tensors.
- B strict RGB ANN twin: BN -> Conv -> ReLU.
- C ANN-native RGB: Conv -> BN -> ReLU.

The sensing modality differs between A and B/C, so describe them as paired event/spiking and frame/ANN perception pipelines rather than a pure neuron-only comparison.

## Limited-data training policy

The six Event-SNNs may be lightly trained on the available DAGr/DSEC fragment to obtain learned activity. Training is fixed-budget and ordinary. Do not force a target activity, rescue dead layers, adapt thresholds after training, or select a model because it happens to produce favorable energy.

The same fragment may be replayed for computational characterization if the result is explicitly labeled **in-sample computational characterization**. Do not report same-data mAP as generalization performance.

## Final benchmark order

1. train six Event-SNN checkpoints;
2. freeze them;
3. run the unified 18-model benchmark;
4. record per-layer spike-health diagnostics without modifying the network;
5. run the robust repeated NVML measurement;
6. run the SANA-FE projection from the recorded trained activity.

## Evidence separation

Keep these columns conceptually separate:

- architecture/memory;
- measured spike activity;
- Chapter-6 analytical energy;
- NVIDIA runtime/board energy;
- SANA-FE neuromorphic projection.

No other external hardware framework is part of v1.0.9.

## SANA-FE reporting

Archive for every SANA-FE run:

- SANA-FE version;
- timing model and timestep count;
- generated proxy architecture YAML;
- per-model projection plan JSON;
- measured activity source;
- physical-core/chip-equivalent estimate;
- raw SANA-FE output-derived metrics;
- the explicit scope note that dense SSD prediction heads are excluded.

The SANA-FE proxy is a layer-aggregated Loihi-like projection, not a functional conversion of the PLIF detector.
