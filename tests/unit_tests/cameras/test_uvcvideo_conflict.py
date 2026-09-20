"""
Detecting the uvcvideo/libk4a conflict — ``viki.cameras.kinect``.

The k4a SDK drives the Kinect colour camera over libusb/usbfs. When the kernel's
``uvcvideo`` also binds its UVC interfaces, pipewire probes the device as a
webcam and the two owners collide on one control endpoint. ViKi cannot fix that
from inside the container (``/sys`` is read-only), so it must at least name it.
"""

from __future__ import annotations

from viki.cameras.kinect import uvcvideo_claimed_kinect_interfaces


def _sysfs(tmp_path, bound: dict[str, tuple[str, str]], extras=()):
    """Build a fake ``uvcvideo`` driver dir + usb devices tree."""
    drv = tmp_path / "uvcvideo"
    drv.mkdir()
    devs = tmp_path / "devices"
    devs.mkdir()
    for iface, (vid, pid) in bound.items():
        (drv / iface).mkdir()
        dev = devs / iface.split(":")[0]
        dev.mkdir(exist_ok=True)
        (dev / "idVendor").write_text(vid + "\n")
        (dev / "idProduct").write_text(pid + "\n")
    for name in extras:  # bind/unbind/module live here too
        (drv / name).touch()
    return str(drv), str(devs)


def test_reports_kinect_colour_interfaces(tmp_path):
    drv, devs = _sysfs(
        tmp_path,
        {"2-3.1:1.0": ("045e", "097d"), "2-3.1:1.1": ("045e", "097d")},
        extras=("bind", "unbind", "module"),
    )
    assert uvcvideo_claimed_kinect_interfaces(drv, devs) == ["2-3.1:1.0", "2-3.1:1.1"]


def test_ignores_other_webcams(tmp_path):
    """An ordinary USB camera on uvcvideo is none of our business."""
    drv, devs = _sysfs(tmp_path, {"1-2:1.0": ("1bcf", "2c99")})
    assert uvcvideo_claimed_kinect_interfaces(drv, devs) == []


def test_ignores_the_kinect_depth_camera(tmp_path):
    """Only the colour camera (097d) is the one uvcvideo fights us over."""
    drv, devs = _sysfs(tmp_path, {"2-3.2:1.0": ("045e", "097c")})
    assert uvcvideo_claimed_kinect_interfaces(drv, devs) == []


def test_no_sysfs_is_not_an_error(tmp_path):
    assert uvcvideo_claimed_kinect_interfaces(str(tmp_path / "nope"), str(tmp_path)) == []
