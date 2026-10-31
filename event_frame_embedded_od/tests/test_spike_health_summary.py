import math
import torch
from torch import nn

from efod.evaluation import spike_activity_sequence
from efod.models.neurons import ParametricLIF


class TinySpikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.act1 = ParametricLIF(init_tau=2.0, threshold=1.0)
        self.act2 = ParametricLIF(init_tau=2.0, threshold=1.0)

    def reset_state(self):
        self.act1.reset_state()
        self.act2.reset_state()

    def forward_raw(self, x):
        # First layer always spikes for x=3; second remains silent for zero input.
        y = self.act1(x)
        z = self.act2(torch.zeros_like(y))
        return z


def test_layer_min_median_max_are_numeric_not_nan():
    model = TinySpikeModel().eval()
    inputs = [torch.full((1, 1, 2, 2), 3.0) for _ in range(3)]
    out = spike_activity_sequence(model, inputs, state_mode='stream')

    assert out['seq_spike_layer_count'] == 2
    assert not math.isnan(out['seq_spike_layer_rate_min'])
    assert not math.isnan(out['seq_spike_layer_rate_median'])
    assert not math.isnan(out['seq_spike_layer_rate_max'])
    assert out['seq_spike_layer_rate_min'] == 0.0
    assert out['seq_spike_layer_rate_max'] > 0.0
