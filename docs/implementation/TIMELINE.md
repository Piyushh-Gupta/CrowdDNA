# Timeline & Capacity Review

## Assumptions
- Velocity is constant.
- No severe architectural pivots required.

## Estimates
- **One Developer**: 24 Weeks (Sequential blocking).
- **Two Developers (Current State)**: 12-14 Weeks (Parallelizing Backend/Frontend once API contracts are set).
- **Three Developers**: 10 Weeks (Diminishing returns due to tight coupling in core ML; only helpful for CI/DevOps parallelization).

## Critical Path
Data Eng -> Graph ML -> API -> SDK/Frontend.
