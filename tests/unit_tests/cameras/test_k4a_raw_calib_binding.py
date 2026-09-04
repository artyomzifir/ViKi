"""The offline depth↔colour path needs two extra libk4a bindings:
``k4a_device_get_raw_calibration`` (record time) and
``k4a_calibration_get_from_raw`` (offline rebuild). Pin their ctypes signatures
so a bad edit fails loudly. Skips where libk4a isn't installed."""

import pytest


@pytest.fixture(scope="module")
def kinect():
    try:
        from viki.cameras import kinect as k
    except OSError:
        pytest.skip("libk4a not installed")
    return k


def test_raw_calibration_bindings_present(kinect):
    lib = kinect._lib
    assert hasattr(lib, "k4a_device_get_raw_calibration")
    assert hasattr(lib, "k4a_calibration_get_from_raw")
    assert hasattr(lib, "k4a_device_get_sync_jack")


def test_get_raw_calibration_signature(kinect):
    fn = kinect._lib.k4a_device_get_raw_calibration
    # (device, uint8* data, size_t* data_size)
    assert len(fn.argtypes) == 3
    assert fn.restype is not None


def test_get_from_raw_signature(kinect):
    fn = kinect._lib.k4a_calibration_get_from_raw
    # (char* raw, size_t size, int depth_mode, int color_res, void* calib_out)
    assert len(fn.argtypes) == 5


def test_get_sync_jack_signature(kinect):
    fn = kinect._lib.k4a_device_get_sync_jack
    assert len(fn.argtypes) == 3


def test_buffer_result_constants(kinect):
    assert (kinect.K4A_BUFFER_RESULT_SUCCEEDED,
            kinect.K4A_BUFFER_RESULT_TOO_SMALL) == (0, 2)


def test_backend_exposes_get_raw_calibration(kinect):
    # method exists on the class and defaults to None before start()
    b = object.__new__(kinect.KinectBackend)
    b._raw_calibration = None
    assert b.get_raw_calibration() is None


def test_backend_itself_refuses_standalone_with_multiple_devices(kinect, monkeypatch):
    from viki.cameras.hw_sync import HardwareSyncError

    monkeypatch.setattr(kinect.KinectBackend, "device_count", staticmethod(lambda: 2))
    backend = kinect.KinectBackend(wired_sync_mode=kinect.K4A_WIRED_SYNC_MODE_STANDALONE)
    with pytest.raises(HardwareSyncError, match="standalone mode is forbidden"):
        backend.start()
