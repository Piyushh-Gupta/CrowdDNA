import pytest
import json
from crowdflow_dna.release.version import VersionManager
from crowdflow_dna.release.manifest import ManifestGenerator
from crowdflow_dna.release.signing import SigningManager
from crowdflow_dna.release.sbom import SBOMGenerator
from crowdflow_dna.release.provenance import ProvenanceGenerator
from crowdflow_dna.release.candidate import CandidateManager
from crowdflow_dna.release.license import LicenseValidator
from crowdflow_dna.release.registry import PublisherRegistry
from crowdflow_dna.release.publishers.github import GitHubPublisher
from crowdflow_dna.release.changelog import ChangelogGenerator
from crowdflow_dna.release.exceptions import VersionError, StateTransitionError, ChecksumError
from crowdflow_dna.release.metadata import ReleaseMetadata, BuildMetadata, DistributionMetadata

def test_semver():
    vm = VersionManager("1.2.3")
    assert vm.bump("patch") == "1.2.4"
    assert vm.bump("minor") == "1.3.0"
    assert vm.bump("major") == "2.0.0"
    assert vm.bump("rc") == "1.2.4-rc.1"
    
    vm_rc = VersionManager("1.2.4-rc.1")
    assert vm_rc.bump("rc") == "1.2.4-rc.2"
    
    with pytest.raises(VersionError):
        VersionManager("invalid-version")

def test_manifest_determinism():
    mg = ManifestGenerator()
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert mg.generate(d1) == mg.generate(d2)
    
    # Test dataclass integration
    rm = ReleaseMetadata(
        version="1.0.0",
        commit="abc",
        build=BuildMetadata(builder_id="1", timestamp="now"),
        artifacts=(),
        distribution=DistributionMetadata(targets=("pypi",))
    )
    res = json.loads(mg.generate(rm))
    assert res["version"] == "1.0.0"
    assert res["distribution"]["targets"] == ["pypi"]

def test_checksum_generation(tmp_path):
    sm = SigningManager(chunk_size=16)
    p = tmp_path / "test.bin"
    p.write_bytes(b"hello world")
    
    cs = sm.generate_checksums(str(p))
    assert "sha256" in cs
    assert "sha512" in cs
    
    with pytest.raises(ChecksumError):
        sm.generate_checksums("does_not_exist.bin")

def test_sbom_generation():
    sg = SBOMGenerator()
    deps = [
        {"name": "requests", "version": "2.28.0", "license": "Apache-2.0"},
        {"name": "requests", "version": "2.28.0", "license": "Apache-2.0"}, # Duplicate
        {"name": "pytest"} # Missing license/version
    ]
    sbom = sg.generate(deps)
    assert len(sbom["components"]) == 2
    assert sbom["components"][0]["name"] == "pytest"
    assert sbom["components"][0]["licenses"][0]["license"]["id"] == "UNKNOWN"

def test_provenance_generation():
    pg = ProvenanceGenerator()
    prov = pg.generate("abc1234", "builder-1", "1.0.0", ["b.tar.gz", "a.whl"])
    assert prov["commit"] == "abc1234"
    assert "timestamp" in prov
    assert prov["artifacts"] == ["a.whl", "b.tar.gz"] # sorted

def test_release_candidates():
    cm = CandidateManager()
    with pytest.raises(StateTransitionError):
        cm.promote()
        
    cm.create()
    assert cm.state == "rc"
    
    with pytest.raises(StateTransitionError):
        cm.create()
        
    cm.promote()
    assert cm.state == "release"
    cm.freeze()
    assert cm.state == "frozen"
    
    cm2 = CandidateManager()
    cm2.create()
    cm2.reject()
    assert cm2.state == "rejected"
    cm2.freeze()

def test_publisher_registry():
    pr = PublisherRegistry()
    pr.register("github", GitHubPublisher())
    assert isinstance(pr.get("github"), GitHubPublisher)

def test_license_validation():
    lv = LicenseValidator(allowlist=["MIT"], denylist=["GPL"])
    deps = [{"name": "pkg1", "license": "MIT"}, {"name": "pkg2", "license": "GPL"}, {"name": "pkg2", "license": "GPL"}]
    report = lv.validate(deps)
    assert not report["compliant"]
    assert len(report["violations"]) == 1 # Deduplicated
    
def test_changelog_generation():
    cg = ChangelogGenerator()
    commits = ["feat: add stuff", "fix: bug", "fix: bug"] # Duplicate
    md = cg.generate(commits)
    assert "## Features" in md
    assert "## Fixes" in md
    assert md.count("fix: bug") == 1 # Deduplicated
