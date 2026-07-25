import pytest
from crowdflow_dna.deployment.state_machine import DeploymentLifecycle, DeploymentState
from crowdflow_dna.deployment.models import ReleaseManifest
from crowdflow_dna.deployment.manager import DeploymentManager

def test_deployment_lifecycle():
    lc = DeploymentLifecycle()
    assert lc.state == DeploymentState.PENDING
    lc.transition(DeploymentState.VALIDATING)
    assert lc.state == DeploymentState.VALIDATING
    
    with pytest.raises(ValueError):
        lc.transition(DeploymentState.ACTIVE) # Invalid jump

def test_release_manifest_json():
    rm = ReleaseManifest(
        release_id="r1",
        release_version="1.0",
        git_commit="abc",
        image_digest="sha256:123",
        build_timestamp="2026",
        frontend_version="1.0",
        api_version="1.0",
        framework_version="1.0",
        reproducibility_manifest_reference="ref1",
        deployment_manifest_reference="ref2",
        artifact_manifest_reference="ref3"
    )
    j = rm.to_json()
    rm2 = ReleaseManifest.from_json(j)
    assert rm2.release_id == "r1"

def test_deployment_manager_lock():
    mgr = DeploymentManager("test.lock")
    mgr.acquire_lock()
    
    mgr2 = DeploymentManager("test.lock")
    with pytest.raises(RuntimeError):
        mgr2.acquire_lock()
        
    mgr.release_lock()
    mgr2.acquire_lock() # Should work now
    mgr2.release_lock()
    
def test_deployment_manager_execution():
    mgr = DeploymentManager("test_exec.lock")
    rm = ReleaseManifest(
        release_id="r1", release_version="1.0", git_commit="abc",
        image_digest="sha256:123", build_timestamp="2026", frontend_version="1.0",
        api_version="1.0", framework_version="1.0", reproducibility_manifest_reference="ref1",
        deployment_manifest_reference="ref2", artifact_manifest_reference="ref3"
    )
    result = mgr.execute_deployment(rm)
    assert result is True
    assert mgr.lifecycle.state == DeploymentState.ACTIVE
