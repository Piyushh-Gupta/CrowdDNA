import platform
import subprocess
import os
import torch
import psutil

from training.reproducibility.metadata import EnvironmentSnapshot

# We'll use a whitelist of environment variables to capture
ENV_WHITELIST = {
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "USER",
    "LOGNAME"
}

def capture_environment_snapshot() -> EnvironmentSnapshot:
    """Captures system, GPU, python, and git environment data securely."""
    # OS
    os_name = platform.system() + " " + platform.release()
    python_version = platform.python_version()
    pytorch_version = torch.__version__
    
    # GPU
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_model = torch.cuda.get_device_name(0)
        vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        try:
            # Simple driver version extraction (Linux typically, might fail on some Windows setups without nvidia-smi in path)
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True, check=True)
            gpu_driver_version = result.stdout.strip()
        except Exception:
            gpu_driver_version = "unknown"
    else:
        cuda_version = "None"
        gpu_model = "None"
        vram_total_mb = 0.0
        gpu_driver_version = "None"
        
    # CPU & RAM
    cpu_model = platform.processor()
    ram_total_mb = psutil.virtual_memory().total / (1024 * 1024)
    
    # Git
    try:
        git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        git_branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        git_is_dirty = bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        git_commit = "unknown"
        git_branch = "unknown"
        git_is_dirty = False
        
    # Pip freeze
    try:
        pip_freeze_raw = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True).stdout.strip().split('\n')
        pip_freeze = tuple(sorted([pkg for pkg in pip_freeze_raw if pkg]))
    except Exception:
        pip_freeze = ("unknown",)
        
    # Env vars
    env_vars = {}
    for var in ENV_WHITELIST:
        if var in os.environ:
            env_vars[var] = os.environ[var]
            
    return EnvironmentSnapshot(
        os_name=os_name,
        python_version=python_version,
        pytorch_version=pytorch_version,
        cuda_version=cuda_version,
        gpu_model=gpu_model,
        gpu_driver_version=gpu_driver_version,
        vram_total_mb=vram_total_mb,
        cpu_model=cpu_model,
        ram_total_mb=ram_total_mb,
        git_commit=git_commit,
        git_branch=git_branch,
        git_is_dirty=git_is_dirty,
        pip_freeze=pip_freeze,
        env_vars=env_vars
    )
