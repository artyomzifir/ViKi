"""
viki.calibration.file
--------------------
File I/O utilities for calibration parameters.

This module handles reading and writing intrinsic and extrinsic parameters
to/from JSON files. The files store a list of entries, each identified by
`device_id`.
"""
import json
import logging
import numpy as np
from viki.config import INTRINSICS_FILENAME, EXTRINSICS_FILENAME
from viki.calibration.models import CalibrationIntrinsics, CalibrationExtrinsics


def write_device_intrinsics(
    device_id: str, intrinsics: CalibrationIntrinsics, file: str = INTRINSICS_FILENAME
):
    """
    Write intrinsic parameters to a JSON file.

    If the file already exists and contains a list, the entry for the given
    `device_id` is updated; otherwise, it is appended.

    Parameters
    ----------
    device_id : str
        Camera identifier.
    intrinsics : CalibrationIntrinsics
        Intrinsic parameters to store.
    file : str
        Path to the JSON file.
    """
    try:
        with open(file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    logging.debug("intrinsics DB: %s", data)
    new_entry = {
        "device_id": device_id,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "dist_coeffs": intrinsics.dist_coeffs.tolist(),
    }

    logging.debug(new_entry)

    for i, entry in enumerate(data):
        if entry.get("device_id") == device_id:
            data[i] = new_entry
            break
    else:
        data.append(new_entry)

    logging.debug("writing intrinsics DB to %s", file)
    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def write_device_extrinsics(
    device_id: str, extrinsics: CalibrationExtrinsics, file: str = EXTRINSICS_FILENAME
):
    """
    Write extrinsic parameters to a JSON file.

    If the file already exists and contains a list, the entry for the given
    `device_id` is updated; otherwise, it is appended.

    Parameters
    ----------
    device_id : str
        Camera identifier.
    extrinsics : CalibrationExtrinsics
        Extrinsic parameters to store.
    file : str
        Path to the JSON file.
    """
    try:
        with open(file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    new_entry = {
        "device_id": device_id,
        "rvec": extrinsics.rvec.tolist(),
        "tvec": extrinsics.tvec.tolist(),
    }

    logging.debug(new_entry)

    for i, entry in enumerate(data):
        if entry.get("device_id") == device_id:
            data[i] = new_entry
            break
    else:
        data.append(new_entry)

    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def read_device_intrinsics(
    device_id: str, file: str = INTRINSICS_FILENAME
) -> CalibrationIntrinsics | None:
    """
    Read intrinsic parameters for a device from a JSON file.

    Parameters
    ----------
    device_id : str
        Camera identifier.
    file : str
        Path to the JSON file.

    Returns
    -------
    Optional[CalibrationIntrinsics]
        Intrinsics if found, else None.
    """
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if not isinstance(data, list):
        return None

    for entry in data:
        if entry.get("device_id") == device_id:
            dist_coeffs = np.array(entry.get("dist_coeffs", [0.0] * 5))

            return CalibrationIntrinsics(
                fx=entry["fx"],
                fy=entry["fy"],
                cx=entry["cx"],
                cy=entry["cy"],
                dist_coeffs=dist_coeffs,
            )

    return None


def read_device_extrinsics(
    device_id: str, file: str = EXTRINSICS_FILENAME
) -> CalibrationExtrinsics | None:
    """
    Read extrinsic parameters for a device from a JSON file.

    Parameters
    ----------
    device_id : str
        Camera identifier.
    file : str
        Path to the JSON file.

    Returns
    -------
    Optional[CalibrationExtrinsics]
        Extrinsics if found, else None.
    """
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if not isinstance(data, list):
        return None

    for entry in data:
        if entry.get("device_id") == device_id:
            rvec = np.array(entry.get("rvec", [0.0] * 3)).flatten()
            tvec = np.array(entry.get("tvec", [0.0] * 3)).flatten()

            return CalibrationExtrinsics(rvec=rvec, tvec=tvec)

    return None
