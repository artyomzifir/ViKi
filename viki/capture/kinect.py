"""
viki.capture.kinect
-------------------
Azure Kinect DK backend using ctypes directly over libk4a.so.
No compilation required — only libk4a.so needs to be installed system-wide.

Tested with libk4a 1.4.1 on Ubuntu 22.04 inside Docker.
"""

from __future__ import annotations

import ctypes
import logging
import cv2
import numpy as np

from .base import CameraBackend, CameraIntrinsics, Frame

logger = logging.getLogger(__name__)


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

K4A_RESULT_SUCCEEDED = 0
K4A_WAIT_RESULT_SUCCEEDED = 0
K4A_WAIT_RESULT_TIMEOUT = 1

K4A_COLOR_RESOLUTION_720P = 1  # 1280x720
K4A_COLOR_RESOLUTION_1080P = 2  # 1920x1080
K4A_COLOR_RESOLUTION_1536P = 4  # 2048x1536

K4A_DEPTH_MODE_NFOV_UNBINNED = 3  # 640x576
K4A_DEPTH_MODE_NFOV_2X2BINNED = 2  # 320x288
K4A_DEPTH_MODE_WFOV_UNBINNED = 5  # 1024x1024
K4A_DEPTH_MODE_WFOV_2X2BINNED = 4  # 512x512

K4A_FRAMES_PER_SECOND_5 = 0
K4A_FRAMES_PER_SECOND_15 = 1
K4A_FRAMES_PER_SECOND_30 = 2

# Wired sync modes — set on K4ADeviceConfig.wired_sync_mode
# STANDALONE  : no sync cable; each device captures independently
# MASTER      : sends sync pulses on SYNC OUT; start this AFTER the subordinate
# SUBORDINATE : receives pulses on SYNC IN; start this FIRST
K4A_WIRED_SYNC_MODE_STANDALONE = 0
K4A_WIRED_SYNC_MODE_MASTER = 1
K4A_WIRED_SYNC_MODE_SUBORDINATE = 2

K4A_IMAGE_FORMAT_COLOR_BGRA32 = 0
K4A_IMAGE_FORMAT_DEPTH16 = 3

K4A_CALIBRATION_TYPE_COLOR = 1
K4A_CALIBRATION_TYPE_DEPTH = 0

# ── ctypes structs ────────────────────────────────────────────────────────────


class K4ADeviceConfig(ctypes.Structure):
    _fields_ = [
        ("color_format", ctypes.c_int),
        ("color_resolution", ctypes.c_int),
        ("depth_mode", ctypes.c_int),
        ("camera_fps", ctypes.c_int),
        ("synchronized_images_only", ctypes.c_bool),
        ("depth_delay_off_color_usec", ctypes.c_int32),
        ("wired_sync_mode", ctypes.c_int),
        ("subordinate_delay_off_master_usec", ctypes.c_uint32),
        ("disable_streaming_indicator", ctypes.c_bool),
    ]


# Opaque handle types
K4ADevice = ctypes.c_void_p
K4ACapture = ctypes.c_void_p
K4AImage = ctypes.c_void_p
K4ACalibration = ctypes.c_void_p

class K4AFloat3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]

class K4AFloat2(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


# ── Function signatures ───────────────────────────────────────────────────────

_lib.k4a_device_get_installed_count.restype = ctypes.c_uint32
_lib.k4a_device_get_installed_count.argtypes = []

_lib.k4a_device_open.restype = ctypes.c_int
_lib.k4a_device_open.argtypes = [ctypes.c_uint32, ctypes.POINTER(K4ADevice)]

_lib.k4a_device_close.restype = None
_lib.k4a_device_close.argtypes = [K4ADevice]

_lib.k4a_device_start_cameras.restype = ctypes.c_int
_lib.k4a_device_start_cameras.argtypes = [K4ADevice, ctypes.POINTER(K4ADeviceConfig)]

_lib.k4a_device_stop_cameras.restype = None
_lib.k4a_device_stop_cameras.argtypes = [K4ADevice]

_lib.k4a_device_get_capture.restype = ctypes.c_int
_lib.k4a_device_get_capture.argtypes = [
    K4ADevice,
    ctypes.POINTER(K4ACapture),
    ctypes.c_int32,
]

_lib.k4a_capture_release.restype = None
_lib.k4a_capture_release.argtypes = [K4ACapture]

_lib.k4a_capture_get_color_image.restype = K4AImage
_lib.k4a_capture_get_color_image.argtypes = [K4ACapture]

_lib.k4a_capture_get_depth_image.restype = K4AImage
_lib.k4a_capture_get_depth_image.argtypes = [K4ACapture]

_lib.k4a_image_get_buffer.restype = ctypes.c_void_p
_lib.k4a_image_get_buffer.argtypes = [K4AImage]

_lib.k4a_image_get_size.restype = ctypes.c_size_t
_lib.k4a_image_get_size.argtypes = [K4AImage]

_lib.k4a_image_get_width_pixels.restype = ctypes.c_int
_lib.k4a_image_get_width_pixels.argtypes = [K4AImage]

_lib.k4a_image_get_height_pixels.restype = ctypes.c_int
_lib.k4a_image_get_height_pixels.argtypes = [K4AImage]

_lib.k4a_image_get_timestamp_usec.restype = ctypes.c_uint64
_lib.k4a_image_get_timestamp_usec.argtypes = [K4AImage]

_lib.k4a_image_release.restype = None
_lib.k4a_image_release.argtypes = [K4AImage]

_lib.k4a_device_get_serialnum.restype = ctypes.c_int
_lib.k4a_device_get_serialnum.argtypes = [
    K4ADevice,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_size_t),
]

# Calibration functions
_lib.k4a_calibration_3d_to_2d.restype = ctypes.c_int
_lib.k4a_calibration_3d_to_2d.argtypes = [
    K4ACalibration,
    ctypes.POINTER(K4AFloat3),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(K4AFloat2),
    ctypes.POINTER(ctypes.c_int),
]

_lib.k4a_calibration_3d_to_3d.restype = ctypes.c_int
_lib.k4a_calibration_3d_to_3d.argtypes = [
    K4ACalibration,
    ctypes.POINTER(K4AFloat3),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(K4AFloat3),
]

_lib.k4a_calibration_2d_to_2d.restype = ctypes.c_int
_lib.k4a_calibration_2d_to_2d.argtypes = [
    K4ACalibration,
    ctypes.POINTER(K4AFloat2),
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(K4AFloat2),
    ctypes.POINTER(ctypes.c_int),
]

_lib.k4a_calibration_2d_to_3d.restype = ctypes.c_int
_lib.k4a_calibration_2d_to_3d.argtypes = [
    K4ACalibration,
    ctypes.POINTER(K4AFloat2),
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(K4AFloat3),
    ctypes.POINTER(ctypes.c_int),
]


# Calibration & transformation
# K4ACalibration = ctypes.c_void_p
K4ATransformation = ctypes.c_void_p

_lib.k4a_device_get_calibration.restype = ctypes.c_int
_lib.k4a_device_get_calibration.argtypes = [
    K4ADevice,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]

_lib.k4a_transformation_create.restype = K4ATransformation
_lib.k4a_transformation_create.argtypes = [ctypes.c_void_p]

_lib.k4a_transformation_destroy.restype = None
_lib.k4a_transformation_destroy.argtypes = [K4ATransformation]

_lib.k4a_image_create.restype = ctypes.c_int
_lib.k4a_image_create.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.POINTER(K4AImage),
]

_lib.k4a_transformation_depth_image_to_color_camera.restype = ctypes.c_int
_lib.k4a_transformation_depth_image_to_color_camera.argtypes = [
    K4ATransformation,
    K4AImage,
    ctypes.POINTER(K4AImage),
]

# ── Resolution maps ───────────────────────────────────────────────────────────

_COLOR_RES_MAP = {
    (1280, 720): K4A_COLOR_RESOLUTION_720P,
    (1920, 1080): K4A_COLOR_RESOLUTION_1080P,
    (2048, 1536): K4A_COLOR_RESOLUTION_1536P,
}

_DEPTH_MODE_MAP = {
    "NFOV_UNBINNED": K4A_DEPTH_MODE_NFOV_UNBINNED,
    "NFOV_2X2BINNED": K4A_DEPTH_MODE_NFOV_2X2BINNED,
    "WFOV_UNBINNED": K4A_DEPTH_MODE_WFOV_UNBINNED,
    "WFOV_2X2BINNED": K4A_DEPTH_MODE_WFOV_2X2BINNED,
}

_FPS_MAP = {
    5: K4A_FRAMES_PER_SECOND_5,
    15: K4A_FRAMES_PER_SECOND_15,
    30: K4A_FRAMES_PER_SECOND_30,
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
    timeout_ms : int
        Frame wait timeout in milliseconds. Default: 5000.
    """

    def __init__(
        self,
        device_index: int = 0,
        color_resolution: tuple[int, int] = (1280, 720),
        depth_mode: str = "NFOV_UNBINNED",
        fps: int = 30,
        timeout_ms: int = 1000,
        align_depth_to_color: bool = False,  # bugged
        wired_sync_mode: int = K4A_WIRED_SYNC_MODE_STANDALONE,
        subordinate_delay_us: int = 0,
        synchronized_images_only: bool = True,
    ) -> None:
        if color_resolution not in _COLOR_RES_MAP:
            raise ValueError(
                f"Unsupported color_resolution {color_resolution}. "
                f"Supported: {list(_COLOR_RES_MAP)}"
            )
        if depth_mode not in _DEPTH_MODE_MAP:
            raise ValueError(
                f"Unknown depth_mode '{depth_mode}'. "
                f"Supported: {list(_DEPTH_MODE_MAP)}"
            )
        if fps not in _FPS_MAP:
            raise ValueError(f"Supported fps: 5, 15, 30. Got: {fps}")

        # WFOV_UNBINNED only supports up to 15 fps
        if depth_mode == "WFOV_UNBINNED" and fps > 15:
            raise ValueError(
                f"WFOV_UNBINNED only supports fps <= 15. Got: {fps}. Use 5 or 15."
            )

        self._device_index = device_index
        self._color_resolution = color_resolution
        self._depth_mode = depth_mode
        self._fps = fps
        self._timeout_ms = timeout_ms
        self._depth_resolution = (640, 576) # Default, updated in start()
        self._align_depth = align_depth_to_color

        self._wired_sync_mode = wired_sync_mode
        self._subordinate_delay_us = subordinate_delay_us
        self._synchronized_images_only = synchronized_images_only
        self._handle: K4ADevice = K4ADevice(None)
        self._transform: K4ATransformation = K4ATransformation(None)
        self._calibration: K4ACalibration = K4ACalibration(None)
        self._calibration_buf: ctypes.Array[ctypes.c_char] | None = None
        self._color_intrinsics: CameraIntrinsics | None = None
        self._depth_intrinsics: CameraIntrinsics | None = None
        self._serial_str: str = f"kinect_{device_index}"
        self._running = False

    # ── CameraBackend interface ───────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return

        # Open device
        res = _lib.k4a_device_open(self._device_index, ctypes.byref(self._handle))
        if res != K4A_RESULT_SUCCEEDED:
            raise RuntimeError(
                f"k4a_device_open failed (result={res}). "
                "Check udev rules and USB permissions."
            )

        # Read serial number
        size = ctypes.c_size_t(64)
        buf = ctypes.create_string_buffer(64)
        _lib.k4a_device_get_serialnum(self._handle, buf, ctypes.byref(size))
        self._serial_str = buf.value.decode(errors="replace")

        # Build config
        config = K4ADeviceConfig(
            color_format=K4A_IMAGE_FORMAT_COLOR_BGRA32,
            color_resolution=_COLOR_RES_MAP[self._color_resolution],
            depth_mode=_DEPTH_MODE_MAP[self._depth_mode],
            camera_fps=_FPS_MAP[self._fps],
            synchronized_images_only=self._synchronized_images_only,
            depth_delay_off_color_usec=0,
            wired_sync_mode=self._wired_sync_mode,
            subordinate_delay_off_master_usec=self._subordinate_delay_us,
            disable_streaming_indicator=False,
        )

        res = _lib.k4a_device_start_cameras(self._handle, ctypes.byref(config))
        if res != K4A_RESULT_SUCCEEDED:
            _lib.k4a_device_close(self._handle)
            raise RuntimeError(f"k4a_device_start_cameras failed (result={res})")

        # Get calibration for reprojection and alignment
        cal_buf = ctypes.create_string_buffer(8192)
        res = _lib.k4a_device_get_calibration(
            self._handle,
            _DEPTH_MODE_MAP[self._depth_mode],
            _COLOR_RES_MAP[self._color_resolution],
            cal_buf,
        )
        if res == K4A_RESULT_SUCCEEDED:
            self._depth_resolution = {
                "NFOV_UNBINNED": (640, 576),
                "NFOV_2X2BINNED": (320, 288),
                "WFOV_UNBINNED": (1024, 1024),
                "WFOV_2X2BINNED": (512, 512),
            }.get(self._depth_mode, (640, 576))
            self._calibration = ctypes.cast(cal_buf, K4ACalibration)
            self._calibration_buf = cal_buf  # keep buffer alive
            # Cache intrinsics once at startup
            self._color_intrinsics = self._get_intrinsics(K4A_CALIBRATION_TYPE_COLOR)
            self._depth_intrinsics = self._get_intrinsics(K4A_CALIBRATION_TYPE_DEPTH)
            if self._color_intrinsics.fx == 0 or self._depth_intrinsics.fx == 0:
                _lib.k4a_device_close(self._handle)
                raise RuntimeError(
                    f"[{self._serial_str}] Failed to extract valid intrinsics from calibration."
                )
        else:
            _lib.k4a_device_close(self._handle)
            raise RuntimeError(f"[{self._serial_str}] Failed to get calibration.")

        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        if self._transform:
            _lib.k4a_transformation_destroy(self._transform)
            self._transform = K4ATransformation(None)
        _lib.k4a_device_stop_cameras(self._handle)
        _lib.k4a_device_close(self._handle)
        self._handle = K4ADevice(None)
        self._running = False

    def get_frame(self) -> Frame:
        if not self._running:
            raise RuntimeError("KinectBackend is not started. Call start() first.")

        capture = K4ACapture(None)
        res = _lib.k4a_device_get_capture(
            self._handle, ctypes.byref(capture), self._timeout_ms
        )
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

            color = self._image_to_numpy_bgr(color_img)
            ts = int(_lib.k4a_image_get_timestamp_usec(color_img))

            if depth_img and (self._align_depth and self._transform):
                # We want BOTH raw and aligned for validation
                raw_depth = self._image_to_numpy_depth(depth_img)
                aligned_depth = self._transform_depth(depth_img, color_img)
            elif depth_img:
                raw_depth = self._image_to_numpy_depth(depth_img)
                aligned_depth = None
                if np.all(raw_depth == 0):
                    _lib.k4a_image_release(color_img)
                    _lib.k4a_image_release(depth_img)
                    raise TimeoutError("Depth frame is all zeros — dropped.")
            else:
                # depth image missing in this capture — return zeros
                h, w = self._depth_resolution[1], self._depth_resolution[0]
                raw_depth = np.zeros((h, w), dtype=np.uint16)
                aligned_depth = None


            _lib.k4a_image_release(color_img)
            if depth_img:
                _lib.k4a_image_release(depth_img)
        finally:
            _lib.k4a_capture_release(capture)

        return Frame(
            color=color,
            depth=raw_depth,
            aligned_depth=aligned_depth,
            timestamp_us=ts,
            device_id=self._serial_str,
            color_intrinsics=self._color_intrinsics,
            depth_intrinsics=self._depth_intrinsics,
        )

    def get_validated_depth(
        self, u: float, v: float, z_est: float, raw_depth: np.ndarray, aligned_depth: np.ndarray | None
    ) -> tuple[float, float, float] | None:
        """
        Simplified depth projection.
        
        Returns: (u_depth, v_depth, final_z) or None.
        """
        res = self.project_color_to_depth(u, v, z_est)
        if res is None:
            return None
        ud, vd = res
        ui, vi = int(round(ud)), int(round(vd))
        h, w = raw_depth.shape[:2]
        if not (0 <= vi < h and 0 <= ui < w):
            return None
        return ud, vd, float(raw_depth[vi, ui])

    def deproject_2d_to_3d(self, u: float, v: float, z: float) -> tuple[float, float, float] | None:
        """
        Deproject a depth pixel (u, v) with depth z (metres) to 3D in depth camera space.
        Uses SDK calibration (handles distortion).
        """
        if not self._calibration:
            return None

        src = K4AFloat2(u, v)
        dst = K4AFloat3()
        valid = ctypes.c_int()

        res = _lib.k4a_calibration_2d_to_3d(
            self._calibration, ctypes.byref(src), z * 1000.0,
            K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH,
            ctypes.byref(dst), ctypes.byref(valid),
        )

        if res == K4A_RESULT_SUCCEEDED and valid.value:
            return dst.x / 1000.0, dst.y / 1000.0, dst.z / 1000.0

        return None

    def project_color_to_depth(self, u: float, v: float, z: float) -> tuple[float, float] | None:
        """Project a color pixel to a depth pixel using SDK calibration (distortion + extrinsics)."""
        if not self._calibration:
            logger.error(f"[{self._serial_str}] project_color_to_depth failed: no calibration handle")
            return None

        src = K4AFloat2(u, v)
        dst = K4AFloat2()
        valid = ctypes.c_int()

        # SDK expects depth in millimetres
        res = _lib.k4a_calibration_2d_to_2d(
            self._calibration, ctypes.byref(src), z * 1000.0,
            K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH,
            ctypes.byref(dst), ctypes.byref(valid),
        )

        if res == K4A_RESULT_SUCCEEDED and valid.value:
            return dst.x, dst.y

        logger.debug(f"[{self._serial_str}] SDK 2d_to_2d result={res}, valid={valid.value} for UV=({u}, {v})")
        return None

    def project_3d_to_2d(self, x: float, y: float, z: float, cam_type: int) -> tuple[float, float] | None:
        """Project a 3D point in camera space to a 2D pixel. Coordinates in metres."""
        if not self._calibration:
            logger.error(f"[{self._serial_str}] project_3d_to_2d failed: no calibration handle")
            return None
        
        # SDK expects millimetres
        p = K4AFloat3(x * 1000, y * 1000, z * 1000)
        pix = K4AFloat2()
        valid = ctypes.c_int()
        
        res = _lib.k4a_calibration_3d_to_2d(
            self._calibration, ctypes.byref(p), cam_type, cam_type, ctypes.byref(pix), ctypes.byref(valid)
        )
        
        if res == K4A_RESULT_SUCCEEDED and valid.value:
            return pix.x, pix.y
        
        logger.debug(f"[{self._serial_str}] SDK project_3d_to_2d result={res}, valid={valid.value} for P=({x}, {y}, {z})")
        return None


    def transform_3d_to_3d(self, x: float, y: float, z: float, src_type: int, dst_type: int) -> tuple[float, float, float] | None:
        """Transform a 3D point from one camera coordinate system to another. Coordinates in metres."""
        if not self._calibration:
            logger.error(f"[{self._serial_str}] transform_3d_to_3d failed: no calibration handle")
            return None
            
        # SDK expects millimetres
        p_src = K4AFloat3(x * 1000, y * 1000, z * 1000)
        p_dst = K4AFloat3()
        
        res = _lib.k4a_calibration_3d_to_3d(
            self._calibration, ctypes.byref(p_src), src_type, dst_type, ctypes.byref(p_dst)
        )
        
        if res == K4A_RESULT_SUCCEEDED:
            return p_dst.x / 1000, p_dst.y / 1000, p_dst.z / 1000
        
        logger.debug(f"[{self._serial_str}] SDK transform_3d_to_3d result={res} for P=({x}, {y}, {z})")
        return None


    def _get_intrinsics(self, cam_type: int) -> CameraIntrinsics:
        """Infer intrinsic parameters using SDK projection."""
        # SDK Projection Fallback
        # Project (0,0,1) to get cx, cy
        p0 = (0.0, 0.0, 1.0)
        pix0 = self.project_3d_to_2d(*p0, cam_type)
        
        # Project (1,0,1) to get fx
        pX = (1.0, 0.0, 1.0)
        pixX = self.project_3d_to_2d(*pX, cam_type)
        
        # Project (0,1,1) to get fy
        pY = (0.0, 1.0, 1.0)
        pixY = self.project_3d_to_2d(*pY, cam_type)
        
        if pix0 is None or pixX is None or pixY is None:
            return CameraIntrinsics(0, 0, 0, 0, 0, 0)
            
        cx, cy = pix0
        fx = pixX[0] - cx
        fy = pixY[1] - cy
        
        w, h = (0, 0)
        if cam_type == K4A_CALIBRATION_TYPE_COLOR:
            w, h = self._color_resolution
        else:
            mode = self._depth_mode
            # Map depth mode to resolution
            res_map = {
                "NFOV_UNBINNED": (640, 576),
                "NFOV_2X2BINNED": (320, 288),
                "WFOV_UNBINNED": (1024, 1024),
                "WFOV_2X2BINNED": (512, 512),
            }
            w, h = res_map.get(mode, (640, 576))
            
        return CameraIntrinsics(fx, fy, cx, cy, w, h)


    @property
    def device_id(self) -> str:
        return self._serial_str

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _transform_depth(self, depth_img: K4AImage, color_img: K4AImage) -> np.ndarray:
        """Transform depth image into color camera space using the SDK (for validation backup)."""
        w = _lib.k4a_image_get_width_pixels(color_img)
        h = _lib.k4a_image_get_height_pixels(color_img)
        dw = _lib.k4a_image_get_width_pixels(depth_img)
        dh = _lib.k4a_image_get_height_pixels(depth_img)
        
        logger.debug(f"[{self._serial_str}] Transforming depth ({dw}x{dh}) to color ({w}x{h}) via SDK")
        
        transformed = K4AImage(None)
        res = _lib.k4a_transformation_depth_image_to_color_camera(
            self._transform, depth_img, ctypes.byref(transformed)
        )
        if res != K4A_RESULT_SUCCEEDED:
            logger.error(f"[{self._serial_str}] k4a_transformation_depth_image_to_color_camera failed (res={res})")
            return self._image_to_numpy_depth(depth_img)
        
        result = self._image_to_numpy_depth(transformed)
        _lib.k4a_image_release(transformed)
        return result


    @staticmethod
    def _image_to_numpy_bgr(img: K4AImage) -> np.ndarray:
        w = _lib.k4a_image_get_width_pixels(img)
        h = _lib.k4a_image_get_height_pixels(img)
        size = _lib.k4a_image_get_size(img)
        buf = _lib.k4a_image_get_buffer(img)
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
    def _image_to_numpy_depth(img: K4AImage) -> np.ndarray:
        w = _lib.k4a_image_get_width_pixels(img)
        h = _lib.k4a_image_get_height_pixels(img)
        size = _lib.k4a_image_get_size(img)
        buf = _lib.k4a_image_get_buffer(img)
        if not buf or size == 0:
            return np.zeros((h, w), dtype=np.uint16)
        raw = ctypes.string_at(buf, size)
        return np.frombuffer(raw, dtype=np.uint16).reshape(h, w).copy()

    # ── Static utils ──────────────────────────────────────────────────────────

    @staticmethod
    def device_count() -> int:
        """Return number of connected Kinect devices."""
        return int(_lib.k4a_device_get_installed_count())
