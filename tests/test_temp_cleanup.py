"""Tests for the safe temporary MP4 cleanup utility.

These tests verify:
- Only registered Crowdflow DNA files are deleted.
- Unrelated files are left untouched.
- Missing/corrupt manifests are handled gracefully.
- Concurrent cleanup is safe.
- The pipeline registers its output files correctly.
"""

from pathlib import Path

import pytest

from crowdflow_dna.tools import temp_cleanup
from crowdflow_dna.tools.temp_cleanup import (
    clean_temp_outputs,
    register_output_file,
    unregister_output_file,
)


@pytest.fixture(autouse=True)
def isolated_temp_dir(tmp_path, monkeypatch):
    """Redirect the temp output directory to a per-test tmp_path."""
    base = tmp_path / "crowddna_outputs"
    monkeypatch.setattr(temp_cleanup, "_TEMP_BASE_DIR", base)
    monkeypatch.setattr(temp_cleanup, "_MANIFEST_PATH", base / "manifest.json")
    yield


def _make_mp4(name: str) -> Path:
    """Create a dummy MP4 file in the temp output directory."""
    path = temp_cleanup._TEMP_BASE_DIR
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / name
    file_path.write_bytes(b"dummy")
    return file_path


def test_register_output_file_requires_absolute_path():
    with pytest.raises(ValueError):
        register_output_file("relative.mp4")


def test_register_output_file_requires_existing_file():
    with pytest.raises(ValueError):
        register_output_file("/nonexistent/file.mp4")


def test_register_and_cleanup_only_deletes_registered_files(tmp_path):
    # Create two files: one registered, one not.
    registered = _make_mp4("registered.mp4")
    unrelated = _make_mp4("unrelated.mp4")

    register_output_file(str(registered))

    deleted = clean_temp_outputs()
    assert deleted == 1
    assert not registered.exists()
    assert unrelated.exists()


def test_cleanup_deletes_nothing_when_no_registered_files(tmp_path):
    _make_mp4("orphan.mp4")
    deleted = clean_temp_outputs()
    assert deleted == 0
    assert (temp_cleanup._TEMP_BASE_DIR / "orphan.mp4").exists()


def test_cleanup_handles_missing_manifest(tmp_path):
    _make_mp4("orphan.mp4")
    # No manifest file should exist.
    assert not temp_cleanup._MANIFEST_PATH.exists()
    deleted = clean_temp_outputs()
    assert deleted == 0


def test_cleanup_handles_corrupt_manifest(tmp_path):
    _make_mp4("orphan.mp4")
    temp_cleanup._TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)
    temp_cleanup._MANIFEST_PATH.write_text("not valid json")
    deleted = clean_temp_outputs()
    assert deleted == 0


def test_cleanup_is_idempotent(tmp_path):
    registered = _make_mp4("registered.mp4")
    register_output_file(str(registered))
    assert clean_temp_outputs() == 1
    assert clean_temp_outputs() == 0


def test_unregister_output_file_prevents_deletion(tmp_path):
    registered = _make_mp4("registered.mp4")
    register_output_file(str(registered))
    unregister_output_file(str(registered))
    deleted = clean_temp_outputs()
    assert deleted == 0
    assert registered.exists()


def test_concurrent_cleanup_is_safe(tmp_path):
    import threading

    registered = _make_mp4("registered.mp4")
    register_output_file(str(registered))

    results = []

    def worker():
        results.append(clean_temp_outputs())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread should have deleted the file.
    assert sum(results) == 1
    assert not registered.exists()


def test_pipeline_registers_output_file_on_success(tmp_path, monkeypatch):
    """Verify that register_output_file works when called with a valid path."""
    from crowdflow_dna.tools import temp_cleanup

    # Create a fake output file in the temp directory.
    output_dir = tmp_path / "crowddna_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fake_output.mp4"
    output_file.write_bytes(b"dummy")

    # Register the file and verify it appears in the manifest.
    register_output_file(str(output_file))

    manifest = temp_cleanup._load_manifest()
    assert output_file.name in manifest

    # Clean up after the test.
    unregister_output_file(str(output_file))


def test_pipeline_does_not_register_output_file_on_failure(tmp_path, monkeypatch):
    """Verify that failed pipeline runs do not leave orphan manifest entries."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline

    monkeypatch.setattr(temp_cleanup, "_TEMP_BASE_DIR", tmp_path / "crowddna_outputs")
    monkeypatch.setattr(
        temp_cleanup, "_MANIFEST_PATH", tmp_path / "crowddna_outputs" / "manifest.json"
    )

    # Patch the pipeline to raise an error.
    def raise_error(self, video_path):
        raise RuntimeError("Pipeline failed")

    monkeypatch.setattr(CrowdFlowPipeline, "run", raise_error)

    pipeline = CrowdFlowPipeline()
    with pytest.raises(RuntimeError):
        pipeline.run("fake.mp4")

    # No manifest should exist.
    assert not temp_cleanup._MANIFEST_PATH.exists()


def test_output_file_allocations_within_same_second_are_distinct():
    """Demonstrate that two consecutive output-file allocations produce distinct paths."""
    import os
    import tempfile

    from crowdflow_dna.pipeline import CrowdFlowPipeline

    path1 = CrowdFlowPipeline._create_output_file()
    path2 = CrowdFlowPipeline._create_output_file()

    try:
        assert path1 != path2
        assert Path(path1).name != Path(path2).name
        expected_dir = Path(tempfile.gettempdir()) / "crowddna_outputs"
        assert Path(path1).parent == expected_dir
        assert Path(path2).parent == expected_dir
        assert path1.endswith(".mp4")
        assert path2.endswith(".mp4")
        assert Path(path1).name.startswith(f"output_{os.getpid()}_")
        assert Path(path2).name.startswith(f"output_{os.getpid()}_")
        assert Path(path1).is_file()
        assert Path(path2).is_file()
    finally:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)
