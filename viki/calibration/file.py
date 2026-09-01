"""
viki.calibration.file
--------------------
Persistence for the **extrinsics** result only (``EXTRINSICS_FILENAME``): a list
of ``{device_id, rvec, tvec}`` entries. Intrinsics are never stored — they come
straight from the camera SDK at the resolution in use.
"""
import json
import logging
import numpy as np
from viki.config import EXTRINSICS_FILENAME
from viki.contracts import CalibrationExtrinsics


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
