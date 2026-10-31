# Release validation

The final standalone reconstruction was validated locally with:

```text
pytest -q
6 passed
```

and:

```text
python scripts/validate_release.py
PASS: all 18 models construct; A/B/C feature interfaces match; Event-SNN core counts match thesis reconstruction.
```

Validated Event-SNN core parameter counts:

- 16-ST-VGG: 125,484
- 32-ST-VGG: 499,788
- 64-ST-VGG: 1,994,892
- ResCat-SNN-16: 216,399
- ResCat-SNN-32: 862,351
- ResCat-SNN-64: 3,442,959

The tests also verify that:

- A contains PLIF neurons; B/C contain none;
- event input is 2 channels and RGB input is 3 channels;
- A/B/C produce identical six SSD feature-map spatial interfaces for each family/width;
- ResCat follows the C -> 2C -> 3C -> 6C -> 6C channel progression;
- all three pipelines produce finite detector training losses on a synthetic target;
- the Chapter-6 analytical energy estimator returns finite positive energy for ANN and SNN paths.

## v1.0.5 validation

- `pytest -q`: **8 passed, 1 skipped** (the skip is the optional local HDF5-plugin environment check).
- `python -m compileall -q src scripts tests`: pass.
- CPU synthetic fast smoke benchmark: all 18 A/B/C configurations complete and produce the new spike-health summary fields.
- GPU/NVML repeated-run protocol cannot be executed in this packaging container because no CUDA/NVML device is attached; the notebook executes it on the user's Colab GPU and saves every repeat for audit.

## v1.0.9 SANA-FE cleanup

- benchmark no longer imports or exposes NeuroBench;
- SpikeSim, SynSense/Samna/Speck, ONNX hardware-export, and generic hardware-preflight scripts are removed;
- dead-layer audit remains diagnostic only;
- SANA-FE projection-plan construction and ResCat branch/join topology are unit-tested without requiring the optional SANA-FE wheel;
- real SANA-FE execution is smoke-tested in Colab after installing `sanafe>=2.2.9,<3`.
