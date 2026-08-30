"""CameraManager.start must restart a running camera when the requested config
differs (regression: it silently no-op'd, so a recording could capture a depth
mode the user never selected)."""

import pytest

from viki.cameras.manager import CameraManager


class _FakeBackend:
    def __init__(self, cfg):
        self._cfg = cfg
        self.started = False

    @property
    def config(self):
        return dict(self._cfg)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def join(self, timeout=8.0):
        pass

    def latest(self):
        return None


@pytest.fixture
def mgr(monkeypatch):
    m = CameraManager()
    made = []

    def _fake_make(device_id, fps, cw, ch, depth_mode, **kw):
        b = _FakeBackend({"color_width": cw, "color_height": ch, "fps": fps,
                          "depth_mode": depth_mode})
        made.append(b)
        return b

    monkeypatch.setattr(CameraManager, "_make_backend", staticmethod(_fake_make))
    # _CameraWorker wraps the backend; make it a thin passthrough
    monkeypatch.setattr("viki.cameras.manager._CameraWorker",
                        lambda backend: type("W", (), {
                            "backend": backend, "start": backend.start,
                            "stop": backend.stop, "join": lambda self, t=8: None,
                            "latest": lambda self: None,
                        })())
    m._made = made
    return m


def test_start_then_unchanged_then_restart(mgr):
    r1 = mgr.start("kinect_0", fps=30, color_width=1280, color_height=720,
                   depth_mode="NFOV_UNBINNED")
    assert r1 == "started" and len(mgr._made) == 1

    r2 = mgr.start("kinect_0", fps=30, color_width=1280, color_height=720,
                   depth_mode="NFOV_UNBINNED")
    assert r2 == "unchanged" and len(mgr._made) == 1  # no new backend

    r3 = mgr.start("kinect_0", fps=30, color_width=1280, color_height=720,
                   depth_mode="WFOV_2X2BINNED")
    assert r3 == "restarted" and len(mgr._made) == 2  # rebuilt with the new mode
    assert mgr._workers["kinect_0"].backend.config["depth_mode"] == "WFOV_2X2BINNED"
