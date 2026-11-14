"""Crowdflow DNA temporary MP4 file management.

Provides two public functions:

* :func:`register_output_file` — record a Crowdflow-generated output file so
  that future cleanup calls know it is safe to delete.
* :func:`clean_temp_outputs` — delete only files that were registered by this
  process, leaving unrelated files (user downloads, other applications, etc.)
  untouched.

A JSON manifest stored next to the output directory tracks which filenames are
owned by Crowdflow DNA.  This keeps the cleanup safe under concurrent use and
across process restarts.
"""

import json
import logging
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Dedicated directory for all Crowdflow DNA temp MP4 outputs.
_TEMP_BASE_DIR = Path(tempfile.gettempdir()) / "crowddna_outputs"

# Manifest that maps each owned filename to its creation metadata.
_MANIFEST_PATH = _TEMP_BASE_DIR / "manifest.json"

# Lock used to serialize manifest reads/writes across threads within this process.
_MANIFEST_LOCK = threading.Lock()


def _ensure_dir() -> None:
    """Ensure the output directory exists."""
    _TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)


def _load_manifest() -> dict:
    """Load the manifest mapping filename -> creation metadata.

    Returns an empty dict if the manifest is missing, corrupt, or empty.
    """
    if not _MANIFEST_PATH.is_file():
        return {}
    try:
        with _MANIFEST_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "files" in data:
                return {
                    entry["name"]: entry
                    for entry in data["files"]
                    if isinstance(entry, dict) and "name" in entry
                }
    except (json.JSONDecodeError, OSError):
        # Corrupt manifest — start fresh rather than fail hard.
        logger.warning("Manifest at %s is corrupt; starting fresh.", _MANIFEST_PATH)
    return {}


def _save_manifest(manifest: dict) -> None:
    """Persist the manifest to disk."""
    _ensure_dir()
    entries = sorted(manifest.values(), key=lambda e: e.get("name", ""))
    with _MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump({"files": entries}, f, indent=2)


def register_output_file(file_path: str) -> None:
    """Register a Crowdflow-generated output file so cleanup can remove it.

    Args:
        file_path: Absolute path to an MP4 file created by the pipeline.

    Raises:
        ValueError: If ``file_path`` is not an absolute path or does not exist.
    """
    if not Path(file_path).is_absolute():
        raise ValueError(f"file_path must be absolute, got: {file_path}")
    if not Path(file_path).exists():
        raise ValueError(f"file_path does not exist: {file_path}")

    entry = {
        "name": Path(file_path).name,
        "path": file_path,
        "created_at": None,  # kept minimal; timestamp could be added later
    }

    with _MANIFEST_LOCK:
        manifest = _load_manifest()
        manifest[entry["name"]] = entry
        _save_manifest(manifest)


def unregister_output_file(file_path: str) -> None:
    """Remove a file path from the manifest without deleting the file.

    Used when a file has been consumed (e.g. downloaded) and should no longer
    be considered a cleanup target.
    """
    with _MANIFEST_LOCK:
        manifest = _load_manifest()
        name = Path(file_path).name
        manifest.pop(name, None)
        _save_manifest(manifest)


def clean_temp_outputs() -> int:
    """Remove temporary MP4 output files that were created by Crowdflow DNA.

    The function:

    1. Loads the manifest of intentionally created filenames.
    2. Scans ``_TEMP_BASE_DIR`` for files matching ``*.mp4``.
    3. Deletes any such file *only* if it appears in the manifest.
    4. Removes the manifest entry for each deleted file.
    5. After the scan, saves the updated manifest.

    Files that are not in the manifest (e.g. user‑downloaded videos, other
    applications' temp files) are left untouched.

    Returns
    -------
    int
        The number of files deleted.
    """
    _ensure_dir()
    with _MANIFEST_LOCK:
        manifest = _load_manifest()

    deleted = 0
    to_remove: set = set()

    # Iterate over all .mp4 files in the dedicated output directory.
    for candidate in sorted(_TEMP_BASE_DIR.glob("*.mp4")):
        name = candidate.name
        if name in manifest:
            # Only delete if it's still on disk and registered.
            try:
                candidate.unlink()
                deleted += 1
                logger.info("Deleted Crowdflow temp output: %s", candidate)
            except OSError as exc:
                logger.warning("Could not delete temp output %s: %s", candidate, exc)
                # Keep the entry so a future run can retry.
                continue
            # Remove entry after successful deletion.
            to_remove.add(name)

    with _MANIFEST_LOCK:
        if to_remove:
            current = _load_manifest()
            for name in to_remove:
                current.pop(name, None)
            _save_manifest(current)

    return deleted
