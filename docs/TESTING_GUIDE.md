# Testing Guide

## Test Suites

1. **Unit Testing**: 
   - When to run: Continuously during development.
   - Purpose: Validates isolated functions and classes using mocks.
2. **Integration Testing**:
   - When to run: Pre-commit and CI.
   - Purpose: Validates API endpoints, database state, and file interactions.
3. **Architecture Validation**:
   - When to run: CI.
   - Purpose: Ensures no circular dependencies or abstraction leaks exist.
4. **Performance Testing**:
   - When to run: Nightly builds.
   - Purpose: Validates latency and throughput baselines.
5. **Documentation Testing**:
   - When to run: Pre-commit.
   - Purpose: Validates cross-references and structural integrity using `test_documentation.py`.
6. **Release Validation**:
   - When to run: Pre-release candidate generation.
   - Purpose: Verifies SBOMs, checksums, and artifact signatures.
