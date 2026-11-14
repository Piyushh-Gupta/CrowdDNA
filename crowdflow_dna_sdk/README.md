# CrowdDNA Python SDK (`crowdflow_dna_sdk`)

Official lightweight Python SDK for integrating with the CrowdDNA Real-Time Video Inference REST API.

## Features

- **Zero Third-Party Runtime Dependencies**: Built entirely on standard library (`urllib.request`, `dataclasses`, `json`).
- **Complete Type Annotations**: Strongly typed models for jobs, timelines, predictions, and metadata.
- **Asynchronous Execution Lifecycle**: Simplified submission, polling, cancellation, and timeout management.
- **Resilient Endpoints**: Automatic route resolution across standard and versioned paths.

## Installation

```bash
pip install ./crowdflow_dna_sdk
```

## Quick Start

```python
from crowdflow_dna_sdk import CrowdDNAClient

client = CrowdDNAClient(base_url="http://localhost:8000")

# Health check
health = client.get_health()
print("Health:", health.status)

# Submit inference job
job = client.submit_inference(video_path="crowd.mp4", enable_calibration=True)
print("Submitted Job ID:", job.job_id)

# Wait for completion
completed_job = client.wait_for_job(job.job_id, timeout_seconds=120.0)
print("Finished! Output video:", completed_job.result.output_video_url)
```
