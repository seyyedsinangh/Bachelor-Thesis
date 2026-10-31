import pytest

def test_dsec_module_registers_hdf5plugin():
    hdf5plugin = pytest.importorskip("hdf5plugin")
    import h5py
    from efod.data import dsec
    assert h5py.h5z.filter_avail(32001)
    dsec._ensure_hdf5_filters()
