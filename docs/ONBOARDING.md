# CrowdDNA Onboarding Guide

Welcome to CrowdDNA! This guide will get you productive in under one hour.

## Repository Overview
CrowdDNA is a platform for crowd simulation and analysis. It is split into a Python backend (ML/Graph Pipelines) and a frontend web application.

## Architecture Overview
Please read `ARCHITECTURE.md` and `docs/ARCHITECTURE_OVERVIEW.md` to understand the system boundaries and design principles.

## Setup
```bash
git clone https://github.com/Piyushh-Gupta/CrowdDNA.git
cd CrowdDNA
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt -r requirements-ci.txt
```

## Environment Variables
Copy the `.env.example` to `.env` and fill in the required development keys.

## Running Tests & Lint
```bash
ruff check .
pytest
```

## First Contribution Walkthrough
1. Find an issue labeled `good first issue`.
2. Branch off `develop`: `git checkout -b feat/my-first-feature`.
3. Make your changes and write tests.
4. Run `ruff check .` and `pytest`.
5. Push your branch and open a PR against `develop`.

## CI
Our CI runs Ruff, Pytest, and Security validations. All checks must pass before merging.

## Debugging & Common Commands
- **Lint Fix:** `ruff check . --fix`
- **Run specific test:** `pytest tests/test_module.py`
