"""Fetch and normalize the five public AllenSDK cache tables."""
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .schema import Metric, Region

SOURCE = "https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-neuropixels/ecephys-cache/"
SESSION_TYPE = "brain_observatory_1.1"
METRICS = f"{SESSION_TYPE}_analysis_metrics.csv"
SOURCE_HASHES = {
    METRICS: "a5335a7ac1e686714b081b9da03be3e632c6cff27cf3d762b5cae7196edb5209",
    "channels.csv": "d2932f1e60f0bd9013a02160f137cb7718654f3ae54eefb1358b7696f3caed98",
    "probes.csv": "f555ce63317d5b6a83bfafd9f7e6a7b7393c0d7416952bafae0fe67412bfb6bf",
    "sessions.csv": "cdd12b8becac84d284f222718a2a94f4155893fb9f26f1a9c725f541acc738f8",
    "units.csv": "1dffbe8dac7f222ec2ef4f7810465ac173a0acf29b44d980274e7a664679f078",
}
QUALITY_COLUMNS = ["quality", "amplitude_cutoff", "presence_ratio", "isi_violations"]
METRIC_COLUMNS = [metric.value for metric in Metric]
DATA_COLUMNS = ["unit_id", "session_id", "specimen_id", "region", *QUALITY_COLUMNS, *METRIC_COLUMNS]


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    """Replace a file atomically after writing to a temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(payload)
            temporary.close()
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


def write_json(path: Path, payload) -> None:
    write_bytes(path, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())


def _ensure_source(path: Path, expected: str) -> None:
    if path.exists():
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch for {path}. Use a clean raw directory.")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            with urlopen(SOURCE + path.name, timeout=60) as response:
                shutil.copyfileobj(response, temporary)
            temporary.close()
            if sha256(temp_path) != expected:
                raise ValueError(f"Downloaded {path.name} differs from the pinned Allen snapshot.")
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


def _read(raw: Path, name: str, columns: list[str], **rename) -> pd.DataFrame:
    return pd.read_csv(raw / name, usecols=columns, float_precision="round_trip").rename(columns=rename)


def normalize(raw: Path) -> pd.DataFrame:
    """Join metrics to unit, channel, probe and session metadata by stable Allen IDs."""
    metrics = _read(raw, METRICS, ["ecephys_unit_id", *METRIC_COLUMNS], ecephys_unit_id="unit_id")
    units = _read(raw, "units.csv", ["id", "ecephys_channel_id", *QUALITY_COLUMNS], id="unit_id")
    channels = _read(raw, "channels.csv", ["id", "ecephys_probe_id", "ecephys_structure_acronym"],
                     id="ecephys_channel_id", ecephys_structure_acronym="region")
    probes = _read(raw, "probes.csv", ["id", "ecephys_session_id"],
                   id="ecephys_probe_id", ecephys_session_id="session_id")
    sessions = _read(raw, "sessions.csv", ["id", "session_type", "specimen_id"], id="session_id")
    data = metrics.merge(units, on="unit_id", how="left", validate="one_to_one")
    for table, key in [(channels, "ecephys_channel_id"), (probes, "ecephys_probe_id"),
                       (sessions, "session_id")]:
        data = data.merge(table, on=key, how="left", validate="many_to_one")
    if not data["session_type"].eq(SESSION_TYPE).all():
        raise ValueError("Metrics include unmatched or unexpected sessions")
    visual = data.loc[data["region"].isin([region.value for region in Region]), DATA_COLUMNS]
    return visual.sort_values("unit_id").reset_index(drop=True)


def prepare(path: Path, raw: Path) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(SOURCE_HASHES)) as workers:
        list(workers.map(lambda item: _ensure_source(raw / item[0], item[1]), SOURCE_HASHES.items()))
    frame = normalize(raw)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode()
    manifest = {
        "dataset": "Allen Brain Observatory Visual Coding Neuropixels",
        "normalization_version": 1,
        "session_type": SESSION_TYPE,
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "rows": len(frame),
        "session_ids": sorted(int(value) for value in frame["session_id"].unique()),
        "sources": {name: {"url": SOURCE + name, "sha256": digest}
                    for name, digest in SOURCE_HASHES.items()},
    }
    write_bytes(path, csv_bytes)
    write_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def load(path: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if sha256(path) != manifest["sha256"]:
        raise ValueError(f"Prepared dataset SHA-256 mismatch for {path}")
    frame = pd.read_csv(path, usecols=DATA_COLUMNS, float_precision="round_trip")
    if len(frame) != manifest["rows"] or not frame["unit_id"].is_unique:
        raise ValueError("Prepared dataset does not match its manifest")
    return frame, manifest
