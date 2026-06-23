"""
viki.capture.kinect
-------------------
Azure Kinect DK backend using ctypes directly over libk4a.so.
No compilation required — only libk4a.so needs to be installed system-wide.

Tested with libk4a 1.4.1 on Ubuntu 22.04 inside Docker.
"""

from __future__ import annotations

import ctypes
import cv2
import ctypes.util
import threading
import time
import numpy as np
from typing import Optional

from .base import CameraBackend, CameraIntrinsics, Frame

# ── Load libk4a ───────────────────────────────────────────────────────────────

def _load_libk4a() -> ctypes.CDLL:
    for name in ("libk4a.so", "libk4a.so.1.4", "libk4a.so.1"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError(
        "libk4a.so not found. Make sure libk4a1.4 is installed:\n"
        "  apt-get install /path/to/libk4a1.4_1.4.1_amd64.deb"
    )

_lib = _load_libk4a()

# ── k4a constants ─────────────────────────────────────────────────────────────

K4A_RESULT_SUCCEEDED        = 0
K4A_WAIT_RESULT_SUCCEEDED   = 0
K4A_WAIT_RESULT_TIMEOUT     = 1

K4A_COLOR_RESOLUTION_720P   = 1   # 1280x720
K4A_COLOR_RESOLUTION_1080P  = 2   # 1920x1080
K4A_COLOR_RESOLUTION_1536P  = 4   # 2048x1536

K4A_DEPTH_MODE_OFF             = 0
K4A_DEPTH_MODE_NFOV_2X2BINNED  = 1  # 320x288
K4A_DEPTH_MODE_NFOV_UNBINNED   = 2  # 640x576
K4A_DEPTH_MODE_WFOV_2X2BINNED  = 3  # 512x512
K4A_DEPTH_MODE_WFOV_UNBINNED   = 4  # 1024x1024
K4A_DEPTH_MODE_PASSIVE_IR      = 5  # 1024x1024 IR only

K4A_FRAMES_PER_SECOND_5   = 0
K4A_FRAMES_PER_SECOND_15  = 1
K4A_FRAMES_PER_SECOND_30  = 2

# Wired sync modes — set on K4ADeviceConfig.wired_sync_mode
# STANDALONE  : no sync cable; each device captures independently
# MASTER      : sends sync pulses on SYNC OUT; start this AFTER the subordinate
# SUBORDINATE : receives pulses on SYNC IN; start this FIRST
K4A_WIRED_SYNC_MODE_STANDALONE   = 0
K4A_WIRED_SYNC_MODE_MASTER       = 1
K4A_WIRED_SYNC_MODE_SUBORDINATE  = 2

K4A_IMAGE_FORMAT_COLOR_MJPG   = 0
K4A_IMAGE_FORMAT_COLOR_NV12   = 1
K4A_IMAGE_FORMAT_COLOR_YUY2   = 2
K4A_IMAGE_FORMAT_COLOR_BGRA32 = 3
K4A_IMAGE_FORMAT_DEPTH16      = 4
K4A_IMAGE_FORMAT_IR16         = 5
K4A_IMAGE_FORMAT_CUSTOM8      = 6
K4A_IMAGE_FORMAT_CUSTOM16     = 7
K4A_IMAGE_FORMAT_CUSTOM       = 8

K4A_CALIBRATION_TYPE_COLOR = 1
K4A_CALIBRATION_TYPE_DEPTH = 0

# ── ctypes structs ────────────────────────────────────────────────────────────

class K4ADeviceConfig(ctypes.Structure):
    _fields_ = [
        ("color_format",           ctypes.c_int),
        ("color_resolution",       ctypes.c_int),
        ("depth_mode",             ctypes.c_int),
        ("camera_fps",             ctypes.c_int),
        ("synchronized_images_only", ctypes.c_bool),
        ("depth_delay_off_color_usec", ctypes.c_int32),
        ("wired_sync_mode",        ctypes.c_int),
        ("subordinate_delay_off_master_usec", ctypes.c_uint32),
        ("disable_streaming_indicator", ctypes.c_bool),
    ]


# Opaque handle types
K4ADevice   = ctypes.c_void_p
K4ACapture  = ctypes.c_void_p
K4AImage    = ctypes.c_void_p


class K4ACalibrationIntrinsicParam(ctypes.Structure):
    _fields_ = [
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("k1", ctypes.c_float),
        ("k2", ctypes.c_float),
        ("k3", ctypes.c_float),
        ("k4", ctypes.c_float),
        ("k5", ctypes.c_float),
        ("k6", ctypes.c_float),
        ("codx", ctypes.c_float),
        ("cody", ctypes.c_float),
        ("p2", ctypes.c_float),
        ("p1", ctypes.c_float),
        ("metric_radius", ctypes.c_float),
    ]


class K4ACalibrationIntrinsicParameters(ctypes.Union):
    _fields_ = [
        ("param", K4ACalibrationIntrinsicParam),
        ("v", ctypes.c_float * 15),
    ]


class K4ACalibrationIntrinsics(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("parameter_count", ctypes.c_uint),
        ("parameters", K4ACalibrationIntrinsicParameters),
    ]


class K4ACalibrationExtrinsics(ctypes.Structure):
    _fields_ = [
        ("rotation", ctypes.c_float * 9),
        ("translation", ctypes.c_float * 3),
    ]


class K4ACalibrationCamera(ctypes.Structure):
    _fields_ = [
        ("extrinsics", K4ACalibrationExtrinsics),
        ("intrinsics", K4ACalibrationIntrinsics),
        ("resolution_width", ctypes.c_int),
        ("resolution_height", ctypes.c_int),
        ("metric_radius", ctypes.c_float),
    ]


class K4ACalibrationHeader(ctypes.Structure):
    _fields_ = [
        ("depth_camera_calibration", K4ACalibrationCamera),
        ("color_camera_calibration", K4ACalibrationCamera),
    ]

# ── Function signatures ───────────────────────────────────────────────────────

_lib.k4a_device_get_installed_count.restype  = ctypes.c_uint32
_lib.k4a_device_get_installed_count.argtypes = []

_lib.k4a_device_open.restype  = ctypes.c_int
_lib.k4a_device_open.argtypes = [ctypes.c_uint32, ctypes.POINTER(K4ADevice)]

_lib.k4a_device_close.restype  = None
_lib.k4a_device_close.argtypes = [K4ADevice]

_lib.k4a_device_start_cameras.restype  = ctypes.c_int
_lib.k4a_device_start_cameras.argtypes = [K4ADevice, ctypes.POINTER(K4ADeviceConfig)]

_lib.k4a_device_stop_cameras.restype  = None
_lib.k4a_device_stop_cameras.argtypes = [K4ADevice]

_lib.k4a_device_get_capture.restype  = ctypes.c_int
_lib.k4a_device_get_capture.argtypes = [K4ADevice, ctypes.POINTER(K4ACapture), ctypes.c_int32]

_lib.k4a_capture_release.restype  = None
_lib.k4a_capture_release.argtypes = [K4ACapture]

_lib.k4a_capture_get_color_image.restype  = K4AImage
_lib.k4a_capture_get_color_image.argtypes = [K4ACapture]

_lib.k4a_capture_get_depth_image.restype  = K4AImage
_lib.k4a_capture_get_depth_image.argtypes = [K4ACapture]

_lib.k4a_image_get_buffer.restype  = ctypes.c_void_p
_lib.k4a_image_get_buffer.argtypes = [K4AImage]

_lib.k4a_image_get_size.restype  = ctypes.c_size_t
_lib.k4a_image_get_size.argtypes = [K4AImage]

_lib.k4a_image_get_width_pixels.restype  = ctypes.c_int
_lib.k4a_image_get_width_pixels.argtypes = [K4AImage]

_lib.k4a_image_get_height_pixels.restype  = ctypes.c_int
_lib.k4a_image_get_height_pixels.argtypes = [K4AImage]

_lib.k4a_image_get_timestamp_usec.restype  = ctypes.c_uint64
_lib.k4a_image_get_timestamp_usec.argtypes = [K4AImage]

_lib.k4a_image_release.restype  = None
_lib.k4a_image_release.argtypes = [K4AImage]

_lib.k4a_device_get_serialnum.restype  = ctypes.c_int
_lib.k4a_device_get_serialnum.argtypes = [
    K4ADevice, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)
]

# Calibration & transformation
K4ACalibration  = ctypes.c_void_p
K4ATransformation = ctypes.c_void_p

_lib.k4a_device_get_calibration.restype  = ctypes.c_int
_lib.k4a_device_get_calibration.argtypes = [
    K4ADevice, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]

_lib.k4a_transformation_create.restype  = K4ATransformation
_lib.k4a_transformation_create.argtypes = [ctypes.c_void_p]

_lib.k4a_transformation_destroy.restype  = None
_lib.k4a_transformation_destroy.argtypes = [K4ATransformation]

_lib.k4a_image_create.restype  = ctypes.c_int
_lib.k4a_image_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(K4AImage)
]

_lib.k4a_transformation_depth_image_to_color_camera.restype  = ctypes.c_int
_lib.k4a_transformation_depth_image_to_color_camera.argtypes = [
    K4ATransformation, K4AImage, K4AImage
]

# ── Resolution maps ───────────────────────────────────────────────────────────

_COLOR_RES_MAP = {
    (1280, 720):  K4A_COLOR_RESOLUTION_720P,
    (1920, 1080): K4A_COLOR_RESOLUTION_1080P,
    (2048, 1536): K4A_COLOR_RESOLUTION_1536P,
}

_DEPTH_MODE_MAP = {
    "NFOV_UNBINNED":  K4A_DEPTH_MODE_NFOV_UNBINNED,
    "NFOV_2X2BINNED": K4A_DEPTH_MODE_NFOV_2X2BINNED,
    "WFOV_UNBINNED":  K4A_DEPTH_MODE_WFOV_UNBINNED,
    "WFOV_2X2BINNED": K4A_DEPTH_MODE_WFOV_2X2BINNED,
}

_DEPTH_MODE_RESOLUTION_MAP = {
    "NFOV_2X2BINNED": (320, 288),
    "NFOV_UNBINNED":  (640, 576),
    "WFOV_2X2BINNED": (512, 512),
    "WFOV_UNBINNED":  (1024, 1024),
}

_FPS_MAP = {
    5:  K4A_FRAMES_PER_SECOND_5,
    15: K4A_FRAMES_PER_SECOND_15,
    30: K4A_FRAMES_PER_SECOND_30,
}

_COLOR_FORMAT_MAP = {
    "MJPG": K4A_IMAGE_FORMAT_COLOR_MJPG,
    "BGRA32": K4A_IMAGE_FORMAT_COLOR_BGRA32,
}


# ── Backend ───────────────────────────────────────────────────────────────────

class KinectBackend(CameraBackend):
    """
    Azure Kinect DK backend via ctypes over libk4a.so.
    No pyk4a required.

    Parameters
    ----------
    device_index : int
        Device index (0 for first device).
    color_resolution : tuple[int, int]
        Supported: (1280,720), (1920,1080), (2048,1536). Default: (1280,720).
    depth_mode : str
        One of: NFOV_UNBINNED, NFOV_2X2BINNED, WFOV_UNBINNED, WFOV_2X2BINNED.
        Default: NFOV_UNBINNED.
    fps : int
        5, 15, or 30. Default: 30.
    color_format : str
        MJPG or BGRA32. MJPG avoids decode/re-encode in preview streams.
        Default: MJPG.
    timeout_ms : int
        Frame wait timeout in milliseconds. Default: 5000.
    """

    def __init__(
        self,
        device_index: int = 0,
        color_resolution: tuple[int, int] = (1280, 720),
        depth_mode: str = "NFOV_UNBINNED",
        fps: int = 30,
        color_format: str = "MJPG",
        timeout_ms: int = 1000,
        align_depth_to_color: bool = False,
        wired_sync_mode: int = K4A_WIRED_SYNC_MODE_STANDALONE,
        subordinate_delay_us: int = 0,
        synchronized_images_only: bool = False,
    ) -> None:
        if color_resolution not in _COLOR_RES_MAP:
            raise ValueError(f"Unsupported color_resolution {color_resolution}. "
                             f"Supported: {list(_COLOR_RES_MAP)}")
        if depth_mode not in _DEPTH_MODE_MAP:
            raise ValueError(f"Unknown depth_mode '{depth_mode}'. "
                             f"Supported: {list(_DEPTH_MODE_MAP)}")
        if fps not in _FPS_MAP:
            raise ValueError(f"Supported fps: 5, 15, 30. Got: {fps}")
        color_format = color_format.upper()
        if color_format not in _COLOR_FORMAT_MAP:
            raise ValueError(f"Unsupported color_format '{color_format}'. "
                             f"Supported: {list(_COLOR_FORMAT_MAP)}")

        # WFOV_UNBINNED only supports up to 15 fps
        if depth_mode == "WFOV_UNBINNED" and fps > 15:
            raise ValueError(
                f"WFOV_UNBINNED only supports fps <= 15. Got: {fps}. Use 5 or 15."
            )

        self._device_index    = device_index
        self._color_resolution = color_resolution
        self._depth_mode      = depth_mode
        self._fps             = fps
        self._color_format    = color_format
        self._timeout_ms      = timeout_ms

        self._align_depth             = align_depth_to_color
        self._wired_sync_mode         = wired_sync_mode
        self._subordinate_delay_us    = subordinate_delay_us
        self._synchronized_images_only = synchronized_images_only
        self._handle: K4ADevice   = K4ADevice(None)
        self._transform: K4ATransformation = K4ATransformation(None)
        self._aligned_depth_image: K4AImage = K4AImage(None)
        self._calibration_buf     = None
        self._color_intrinsics: Optional[CameraIntrinsics] = None
        self._transform_lock      = threading.Lock()
        self._calibration_available = False
        self._serial_str: str     = f"kinect_{device_index}"
        self._running             = False
        self._logged_first_frame  = False
        self._warned_transform_failure = False
        self._frame_count         = 0
        self._profile_capture_ms  = 0.0
        self._profile_color_ms    = 0.0
        self._profile_align_ms    = 0.0
        self._profile_depth_ms    = 0.0
        self._profile_total_ms    = 0.0

    # ── CameraBackend interface ───────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return

        # Open device
        res = self._open_device_with_retries()
        if res != K4A_RESULT_SUCCEEDED:
            raise RuntimeError(
                f"k4a_device_open failed for kinect_{self._device_index} "
                f"(result={res}). If the SDK log above contains "
                "cJSON_Parse/calibration_create, libk4a could not read the "
                "device factory calibration; try replugging the camera, using "
                "a different USB3 port/cable, power-cycling the device, or "
                "checking it with k4aviewer/k4arecorder. Also verify udev and "
                "Docker USB permissions."
            )

        # Read serial number
        size = ctypes.c_size_t(64)
        buf  = ctypes.create_string_buffer(64)
        _lib.k4a_device_get_serialnum(self._handle, buf, ctypes.byref(size))
        self._serial_str = buf.value.decode(errors="replace")
        self._logged_first_frame = False
        self._warned_transform_failure = False

        # Build config
        sdk_depth_mode = _DEPTH_MODE_MAP[self._depth_mode]
        config = K4ADeviceConfig(
            color_format           = _COLOR_FORMAT_MAP[self._color_format],
            color_resolution       = _COLOR_RES_MAP[self._color_resolution],
            depth_mode             = sdk_depth_mode,
            camera_fps             = _FPS_MAP[self._fps],
            synchronized_images_only = self._synchronized_images_only,
            depth_delay_off_color_usec = 0,
            wired_sync_mode        = self._wired_sync_mode,
            subordinate_delay_off_master_usec = self._subordinate_delay_us,
            disable_streaming_indicator = False,
        )

        res = _lib.k4a_device_start_cameras(self._handle, ctypes.byref(config))
        if res != K4A_RESULT_SUCCEEDED:
            _lib.k4a_device_close(self._handle)
            raise RuntimeError(f"k4a_device_start_cameras failed (result={res})")

        # Prepare factory calibration for snapshot alignment. The expensive
        # depth->color transform is still only run per frame when explicitly
        # requested by align_depth_to_color=True.
        try:
            self._ensure_transformation(required=self._align_depth)
            if self._align_depth:
                self._aligned_depth_image = self._create_aligned_depth_image()
        except Exception:
            _lib.k4a_device_stop_cameras(self._handle)
            _lib.k4a_device_close(self._handle)
            self._handle = K4ADevice(None)
            raise

        self._running = True
        print(
            f"[kinect:{self._device_index}] started device_id=kinect_{self._device_index} "
            f"serial={self._serial_str} "
            f"color={self._color_resolution[0]}x{self._color_resolution[1]} "
            f"color_format={self._color_format} requested_depth_mode={self._depth_mode} "
            f"sdk_depth_mode={sdk_depth_mode} "
            f"expected_raw_depth={self.expected_raw_depth_resolution} fps={self._fps} "
            f"sync_only={self._synchronized_images_only} "
            f"align_depth_to_color={self._align_depth}"
        )

    def stop(self) -> None:
        if not self._running:
            return
        with self._transform_lock:
            if self._aligned_depth_image:
                _lib.k4a_image_release(self._aligned_depth_image)
                self._aligned_depth_image = K4AImage(None)
            if self._transform:
                _lib.k4a_transformation_destroy(self._transform)
                self._transform = K4ATransformation(None)
            self._calibration_buf = None
            self._color_intrinsics = None
            self._calibration_available = False
        _lib.k4a_device_stop_cameras(self._handle)
        _lib.k4a_device_close(self._handle)
        self._handle  = K4ADevice(None)
        self._running = False
        time.sleep(2.0)  # give USB time to fully release before next open

    def get_frame(self) -> Frame:
        if not self._running:
            raise RuntimeError("KinectBackend is not started. Call start() first.")

        total_t0 = time.perf_counter()
        capture = K4ACapture(None)
        capture_t0 = time.perf_counter()
        res = _lib.k4a_device_get_capture(
            self._handle, ctypes.byref(capture), self._timeout_ms
        )
        capture_ms = (time.perf_counter() - capture_t0) * 1000
        if res == K4A_WAIT_RESULT_TIMEOUT:
            raise TimeoutError("Kinect capture timed out.")
        if res != K4A_WAIT_RESULT_SUCCEEDED:
            raise RuntimeError(f"k4a_device_get_capture failed (result={res})")

        try:
            color_img = _lib.k4a_capture_get_color_image(capture)
            depth_img = _lib.k4a_capture_get_depth_image(capture)

            if not color_img:
                if depth_img:
                    _lib.k4a_image_release(depth_img)
                raise TimeoutError("Color image is NULL in capture — frame dropped.")

            color_t0 = time.perf_counter()
            color_w = _lib.k4a_image_get_width_pixels(color_img)
            color_h = _lib.k4a_image_get_height_pixels(color_img)
            color_shape = (color_h, color_w, 3)
            color_jpeg = None
            if self._color_format == "MJPG":
                color = np.empty((0, 0, 3), dtype=np.uint8)
                color_jpeg = self._image_to_bytes(color_img)
                if not color_jpeg:
                    color = np.zeros(color_shape, dtype=np.uint8)
                    color_jpeg = None
            else:
                color = self._image_to_numpy_bgr(color_img)
            color_ms = (time.perf_counter() - color_t0) * 1000
            ts    = int(_lib.k4a_image_get_timestamp_usec(color_img))
            raw_depth_shape = None

            align_ms = 0.0
            depth_ms = 0.0
            depth_t0 = time.perf_counter()
            raw_depth_for_frame = None
            depth_is_aligned = False
            if depth_img and (self._align_depth and self._transform):
                raw_depth_shape = (
                    _lib.k4a_image_get_height_pixels(depth_img),
                    _lib.k4a_image_get_width_pixels(depth_img),
                )
                raw_copy_t0 = time.perf_counter()
                raw_depth_for_frame = self._image_to_numpy_depth(depth_img)
                depth_ms += (time.perf_counter() - raw_copy_t0) * 1000
                depth, align_ms, copy_ms = self._transform_depth(depth_img, color_img)
                depth_ms += copy_ms
                depth_is_aligned = True
            elif depth_img:
                depth = self._image_to_numpy_depth(depth_img)
                raw_depth_shape = depth.shape
            else:
                # depth image missing in this capture — return zeros
                h, w = self._color_resolution[1], self._color_resolution[0]
                depth = np.zeros((h, w), dtype=np.uint16)
            if not (depth_img and (self._align_depth and self._transform)):
                depth_ms = (time.perf_counter() - depth_t0) * 1000

            if not self._logged_first_frame:
                print(
                    f"[kinect:{self._device_index}] first frame serial={self._serial_str} "
                    f"color_shape={color_shape} color_jpeg={color_jpeg is not None} "
                    f"raw_depth_shape={raw_depth_shape} depth_shape={depth.shape} "
                    f"align_depth_to_color={self._align_depth} depth_is_aligned={depth_is_aligned}"
                )
                self._logged_first_frame = True

            _lib.k4a_image_release(color_img)
            if depth_img:
                _lib.k4a_image_release(depth_img)
        finally:
            _lib.k4a_capture_release(capture)

        total_ms = (time.perf_counter() - total_t0) * 1000
        self._record_profile(capture_ms, color_ms, align_ms, depth_ms, total_ms)

        return Frame(
            color            = color,
            depth            = depth,
            timestamp_us     = ts,
            device_id        = self._serial_str,
            color_intrinsics = self._color_intrinsics,
            depth_intrinsics = None,
            color_jpeg       = color_jpeg,
            color_shape      = color_shape,
            raw_depth        = raw_depth_for_frame,
            depth_is_aligned = depth_is_aligned,
        )

    @property
    def device_id(self) -> str:
        return self._serial_str

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _open_device_with_retries(self, attempts: int = 3) -> int:
        last_res = K4A_RESULT_SUCCEEDED
        for attempt in range(1, attempts + 1):
            self._handle = K4ADevice(None)
            last_res = _lib.k4a_device_open(self._device_index, ctypes.byref(self._handle))
            if last_res == K4A_RESULT_SUCCEEDED:
                if attempt > 1:
                    print(
                        f"[kinect:{self._device_index}] k4a_device_open succeeded "
                        f"after {attempt} attempts"
                    )
                return last_res

            if attempt < attempts:
                print(
                    f"[kinect:{self._device_index}] k4a_device_open failed "
                    f"(result={last_res}, attempt={attempt}/{attempts}); retrying..."
                )
                time.sleep(0.8 * attempt)
            else:
                print(
                    f"[kinect:{self._device_index}] k4a_device_open failed "
                    f"(result={last_res}, attempt={attempt}/{attempts}); giving up"
                )

        return last_res

    def _ensure_transformation(self, required: bool = True) -> bool:
        with self._transform_lock:
            if self._transform:
                if self._color_intrinsics is None and self._calibration_buf is not None:
                    self._extract_color_intrinsics()
                return True

            self._calibration_buf = ctypes.create_string_buffer(4096)
            res = _lib.k4a_device_get_calibration(
                self._handle,
                _DEPTH_MODE_MAP[self._depth_mode],
                _COLOR_RES_MAP[self._color_resolution],
                self._calibration_buf,
            )
            if res != K4A_RESULT_SUCCEEDED:
                self._calibration_available = False
                message = (
                    "k4a_device_get_calibration failed "
                    f"(result={res}, align_depth_to_color={self._align_depth})"
                )
                if required:
                    raise RuntimeError(message)
                print(f"[kinect:{self._device_index}] {message}; snapshots cannot align depth")
                return False

            self._calibration_available = True
            self._extract_color_intrinsics()
            self._transform = _lib.k4a_transformation_create(self._calibration_buf)
            if not self._transform:
                message = "k4a_transformation_create failed"
                if required:
                    raise RuntimeError(message)
                print(f"[kinect:{self._device_index}] {message}; snapshots cannot align depth")
                return False

            return True

    def _extract_color_intrinsics(self) -> Optional[CameraIntrinsics]:
        if self._calibration_buf is None:
            self._color_intrinsics = None
            return None

        calibration = ctypes.cast(
            self._calibration_buf,
            ctypes.POINTER(K4ACalibrationHeader),
        ).contents
        color_calibration = calibration.color_camera_calibration
        params = color_calibration.intrinsics.parameters.param
        width = int(color_calibration.resolution_width) or self._color_resolution[0]
        height = int(color_calibration.resolution_height) or self._color_resolution[1]
        dist_coeffs = np.array(
            [
                params.k1,
                params.k2,
                params.p1,
                params.p2,
                params.k3,
                params.k4,
                params.k5,
                params.k6,
            ],
            dtype=np.float64,
        )
        intrinsics = CameraIntrinsics(
            fx=float(params.fx),
            fy=float(params.fy),
            cx=float(params.cx),
            cy=float(params.cy),
            width=width,
            height=height,
            dist_coeffs=dist_coeffs,
        )
        self._color_intrinsics = intrinsics
        print(
            f"[kinect:{self._device_index}] color intrinsics serial={self._serial_str} "
            f"fx={intrinsics.fx:.3f} fy={intrinsics.fy:.3f} "
            f"cx={intrinsics.cx:.3f} cy={intrinsics.cy:.3f} "
            f"size={intrinsics.width}x{intrinsics.height} "
            f"dist_coeffs={len(intrinsics.dist_coeffs)}"
        )
        return intrinsics

    def _create_aligned_depth_image(self) -> K4AImage:
        w, h = self._color_resolution
        stride = w * 2
        image = K4AImage(None)
        res = _lib.k4a_image_create(
            K4A_IMAGE_FORMAT_DEPTH16, w, h, stride,
            ctypes.byref(image)
        )
        if res != K4A_RESULT_SUCCEEDED:
            raise RuntimeError(
                "k4a_image_create failed for aligned depth output "
                f"(result={res}, width={w}, height={h}, stride={stride})"
            )
        return image

    def _create_depth_image_from_numpy(self, depth: np.ndarray) -> K4AImage:
        if depth.dtype != np.uint16:
            raise ValueError(f"Expected uint16 depth, got {depth.dtype}")
        contiguous = np.ascontiguousarray(depth)
        h, w = contiguous.shape[:2]
        stride = w * 2
        image = K4AImage(None)
        res = _lib.k4a_image_create(
            K4A_IMAGE_FORMAT_DEPTH16, w, h, stride,
            ctypes.byref(image)
        )
        if res != K4A_RESULT_SUCCEEDED:
            raise RuntimeError(
                "k4a_image_create failed for raw depth input "
                f"(result={res}, width={w}, height={h}, stride={stride})"
            )
        buf = _lib.k4a_image_get_buffer(image)
        if not buf:
            _lib.k4a_image_release(image)
            raise RuntimeError("k4a_image_get_buffer returned NULL for raw depth input")
        ctypes.memmove(buf, contiguous.ctypes.data, contiguous.nbytes)
        return image

    def _transform_depth(self, depth_img: K4AImage, color_img: K4AImage) -> tuple[np.ndarray, float, float]:
        """Transform depth image into color camera space."""
        w = _lib.k4a_image_get_width_pixels(color_img)
        h = _lib.k4a_image_get_height_pixels(color_img)

        if not self._aligned_depth_image:
            self._warn_transform_fallback("aligned depth output image is not allocated")
            return np.zeros((h, w), dtype=np.uint16), 0.0, 0.0

        align_t0 = time.perf_counter()
        with self._transform_lock:
            res = _lib.k4a_transformation_depth_image_to_color_camera(
                self._transform, depth_img, self._aligned_depth_image
            )
            align_ms = (time.perf_counter() - align_t0) * 1000
            if res != K4A_RESULT_SUCCEEDED:
                self._warn_transform_fallback(
                    "k4a_transformation_depth_image_to_color_camera failed "
                    f"(result={res})"
                )
                return np.zeros((h, w), dtype=np.uint16), align_ms, 0.0

            copy_t0 = time.perf_counter()
            depth = self._image_to_numpy_depth(self._aligned_depth_image)
            copy_ms = (time.perf_counter() - copy_t0) * 1000
        return depth, align_ms, copy_ms

    def align_depth_snapshot(self, raw_depth: np.ndarray) -> np.ndarray:
        """One-shot raw-depth to color-camera alignment for snapshot/debug use."""
        if not self._running:
            raise RuntimeError("KinectBackend is not started. Call start() first.")
        self._ensure_transformation(required=True)

        input_img = self._create_depth_image_from_numpy(raw_depth)
        output_img = self._create_aligned_depth_image()
        try:
            with self._transform_lock:
                res = _lib.k4a_transformation_depth_image_to_color_camera(
                    self._transform, input_img, output_img
                )
                if res != K4A_RESULT_SUCCEEDED:
                    raise RuntimeError(
                        "k4a_transformation_depth_image_to_color_camera failed "
                        f"for snapshot (result={res})"
                    )
                return self._image_to_numpy_depth(output_img)
        finally:
            _lib.k4a_image_release(input_img)
            _lib.k4a_image_release(output_img)

    def snapshot_calibration_summary(self) -> dict:
        return {
            "factory_calibration_available": bool(self._calibration_available),
            "has_transformation_handle": bool(self._transform),
            "color_intrinsics_available": self._color_intrinsics is not None,
        }

    def _record_profile(
        self,
        capture_ms: float,
        color_ms: float,
        align_ms: float,
        depth_ms: float,
        total_ms: float,
    ) -> None:
        self._frame_count += 1
        self._profile_capture_ms += capture_ms
        self._profile_color_ms += color_ms
        self._profile_align_ms += align_ms
        self._profile_depth_ms += depth_ms
        self._profile_total_ms += total_ms
        if self._frame_count % 60 != 0:
            return

        n = 60.0
        print(
            f"[kinect:{self._device_index}] profile serial={self._serial_str} "
            f"get_capture={self._profile_capture_ms / n:.1f}ms "
            f"color={self._profile_color_ms / n:.1f}ms "
            f"align={self._profile_align_ms / n:.1f}ms "
            f"depth_copy={self._profile_depth_ms / n:.1f}ms "
            f"total={self._profile_total_ms / n:.1f}ms"
        )
        self._profile_capture_ms = 0.0
        self._profile_color_ms = 0.0
        self._profile_align_ms = 0.0
        self._profile_depth_ms = 0.0
        self._profile_total_ms = 0.0

    def _warn_transform_fallback(self, message: str) -> None:
        if self._warned_transform_failure:
            return
        self._warned_transform_failure = True
        print(
            f"[kinect:{self._device_index}] {message}; returning zero depth in "
            f"color geometry because align_depth_to_color={self._align_depth}"
        )

    @staticmethod
    def _image_to_numpy_bgr(img: K4AImage) -> np.ndarray:
        w    = _lib.k4a_image_get_width_pixels(img)
        h    = _lib.k4a_image_get_height_pixels(img)
        size = _lib.k4a_image_get_size(img)
        buf  = _lib.k4a_image_get_buffer(img)
        if not buf or size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        raw = ctypes.string_at(buf, size)
        expected = h * w * 4
        if len(raw) == expected:
            # BGRA32 — direct reshape
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4).copy()
            return arr[:, :, :3]
        else:
            # Compressed (MJPEG) — decode via OpenCV
            arr = np.frombuffer(raw, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            return decoded

    @staticmethod
    def _image_to_bytes(img: K4AImage) -> bytes:
        size = _lib.k4a_image_get_size(img)
        buf  = _lib.k4a_image_get_buffer(img)
        if not buf or size == 0:
            return b""
        return ctypes.string_at(buf, size)

    @staticmethod
    def _image_to_numpy_depth(img: K4AImage) -> np.ndarray:
        w    = _lib.k4a_image_get_width_pixels(img)
        h    = _lib.k4a_image_get_height_pixels(img)
        size = _lib.k4a_image_get_size(img)
        buf  = _lib.k4a_image_get_buffer(img)
        if not buf or size == 0:
            return np.zeros((h, w), dtype=np.uint16)
        raw = ctypes.string_at(buf, size)
        return np.frombuffer(raw, dtype=np.uint16).reshape(h, w).copy()

    # ── Static utils ──────────────────────────────────────────────────────────

    @staticmethod
    def device_count() -> int:
        """Return number of connected Kinect devices."""
        return int(_lib.k4a_device_get_installed_count())

    @property
    def serial_number(self) -> str:
        return self._serial_str

    @property
    def align_depth_to_color(self) -> bool:
        return bool(self._align_depth)

    @property
    def color_format(self) -> str:
        return self._color_format

    @property
    def color_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._color_intrinsics

    @property
    def depth_mode(self) -> str:
        return self._depth_mode

    @property
    def sdk_depth_mode(self) -> int:
        return _DEPTH_MODE_MAP[self._depth_mode]

    @property
    def expected_raw_depth_resolution(self) -> tuple[int, int] | None:
        return _DEPTH_MODE_RESOLUTION_MAP.get(self._depth_mode)

    @property
    def color_resolution(self) -> tuple[int, int]:
        return self._color_resolution
