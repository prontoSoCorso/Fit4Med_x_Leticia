"""Read Xsens MVN .mvnx files with the Python standard library only.

An .mvnx file is XML: a header (who, when, which body segments, sensors and joints)
followed by one <frame> per time step. Only 'normal' frames are kept: the first frames
of a file are calibration poses, not movement.
"""
import csv
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

XYZ = ["x", "y", "z"]
QUATERNION = ["q0", "q1", "q2", "q3"]  # an orientation in 3D, written as 4 numbers
# Each group of numbers in a frame: which list of names it belongs to, and its axes (Xsens MVNX format)
GROUPS = {
    "orientation": ("segments", QUATERNION), "position": ("segments", XYZ),
    "velocity": ("segments", XYZ), "acceleration": ("segments", XYZ),
    "angularVelocity": ("segments", XYZ), "angularAcceleration": ("segments", XYZ),
    "footContacts": ("footContacts", [""]),  # 1 = that part of the foot touches the ground
    "sensorFreeAcceleration": ("sensors", XYZ), "sensorMagneticField": ("sensors", XYZ),
    "sensorOrientation": ("sensors", QUATERNION),
    "jointAngle": ("joints", XYZ), "jointAngleXZY": ("joints", XYZ),
    "jointAngleErgo": ("ergoJoints", XYZ), "jointAngleErgoXZY": ("ergoJoints", XYZ),
    "centerOfMass": ("centerOfMass", XYZ),
}
# Header lists that hold those names: list tag -> name used above
NAME_LISTS = {"segments": "segments", "sensors": "sensors", "joints": "joints",
              "ergonomicJointAngles": "ergoJoints", "footContactDefinition": "footContacts"}
# Original file names look like <subject>_<test>_<timepoint>-<take>, e.g. P01_10MWT_T0-004
FILE_NAME = re.compile(r"(?P<subject>.+)_(?P<test>[^_]+)_(?P<timepoint>[^_]+)-(?P<take>\d+)$")


def local_tag(element):
    return element.tag.split("}")[-1]  # drop the XML namespace


def column_names(frame_values, names):
    """One readable name per number, e.g. jointAngle_jRightKnee_z; unknown groups get numbers."""
    columns = []
    for group, values in frame_values.items():
        owner, axes = GROUPS.get(group, (None, []))
        named = [f"{group}_{name}_{axis}".rstrip("_") for name in names.get(owner, []) for axis in axes]
        columns += named if len(named) == len(values) else [f"{group}_{i}" for i in range(len(values))]
    return columns


def mvnx_to_csv(mvnx_path, csv_path):
    """Write the 'normal' frames of one .mvnx file to a CSV: time_s plus one column per signal.

    Reads the file piece by piece, so long recordings do not fill the memory.
    Returns the header attributes and the number of frames written.
    """
    header, columns, n_frames = {}, None, 0
    names = {"centerOfMass": ["position", "velocity", "acceleration"]}
    with open(csv_path, "w", newline="") as out:
        writer = csv.writer(out)
        for event, element in ET.iterparse(mvnx_path, events=("start", "end")):
            tag = local_tag(element)
            if event == "start" and tag == "subject":
                header = dict(element.attrib)
            elif event == "end" and tag in NAME_LISTS:
                names[NAME_LISTS[tag]] = [item.get("label") for item in element]
            elif event == "end" and tag == "frame":
                if element.get("type") == "normal":
                    values = {local_tag(child): (child.text or "").split() for child in element}
                    if columns is None:
                        columns = column_names(values, names)
                        writer.writerow(["time_s"] + columns)
                    row = [int(element.get("time")) / 1000] + [v for group in values.values() for v in group]
                    if len(row) != len(columns) + 1:
                        raise ValueError(f"{mvnx_path}: frame {element.get('index')} has {len(row) - 1} "
                                         f"values instead of {len(columns)}. Is the file complete?")
                    writer.writerow(row)
                    n_frames += 1
                element.clear()  # this frame is written: free its memory
    if n_frames == 0:
        raise ValueError(f"{mvnx_path}: no 'normal' frames (only calibration poses?): is the recording empty?")
    return header, n_frames


def recording_date(header):
    """Recording start on the lab clock, e.g. 2025-04-07T10:22:44.560 (raw text if unusual)."""
    raw = header.get("recDate", "")
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S.%f %Y").isoformat(timespec="milliseconds")
    except ValueError:
        return raw


def recording_info(header, mvnx_path):
    """Subject, test, timepoint and take from the header's original file name (disk name as fallback)."""
    original = re.split(r"[\\/]", header.get("originalFilename", ""))[-1]  # Windows path -> file name
    for name in (original, Path(mvnx_path).name):
        match = FILE_NAME.match(name.rsplit(".", 1)[0])
        if match:
            return {"subject": match["subject"], "test": match["test"],
                    "timepoint": match["timepoint"].lower(), "take": int(match["take"]),
                    "rec_date": recording_date(header), "frame_rate": float(header.get("frameRate", "nan"))}
    raise ValueError(f"{mvnx_path}: cannot read subject, test, timepoint and take from '{original}' "
                     "or from the file name. Expected a name like P01_10MWT_T0-001.mvnx")
