import math
from efod.evaluation.gpu_energy import _summary_ci, _integrate_power


def test_summary_ci_basic():
    s = _summary_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s['n'] == 5
    assert abs(s['mean'] - 3.0) < 1e-12
    assert s['std'] > 0
    assert s['ci95_low'] < s['mean'] < s['ci95_high']


def test_integrate_constant_power():
    e = _integrate_power([(0.2, 10.0), (0.5, 10.0), (0.8, 10.0)], 0.0, 1.0)
    assert abs(e - 10.0) < 1e-9
