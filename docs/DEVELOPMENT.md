# Local Development

## Environment Setup
1. Clone the repository.
2. Install Python 3.11.
3. Install dependencies: `pip install -r requirements.txt -r requirements-ci.txt`.

## Build
The backend requires no compilation. The frontend builds via `npm run build` inside `frontend/`.

## Test
Execute `python -m pytest tests/` to run all validation suites.

## Run
Use `docker-compose up` inside `deployment/docker/` to spin up the entire stack.
