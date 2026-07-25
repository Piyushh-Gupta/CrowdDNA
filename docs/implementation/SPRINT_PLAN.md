# Sprint Plan

## Sprint 1 (I1: Data Engineering)
- **Deliverables**: Dataset Loaders, Preprocessing pipelines.
- **Review Goal**: Validate CSV loader robustness.
- **Merge Goal**: Merge data loaders into `develop`.
- **Demo Goal**: Terminal output of a normalized trajectory.
- **Definition of Done**: Tests pass, DoR met, Code Reviewed.
- **Exit Criteria**: Downstream Graph ML can consume the data format.

## Sprint 2 (I2: Graph ML)
- **Deliverables**: Training loops, GAT models.
- **Review Goal**: Validate memory limits in training loop.
- **Merge Goal**: Merge ML pipeline.
- **Demo Goal**: Loss curve graph generation.
- **Definition of Done**: Tests pass, DoR met.
- **Exit Criteria**: Model checkpoint successfully exported.

## Sprint 3 (I3: Core API)
- **Deliverables**: FastAPI endpoints for Inference.
- **Review Goal**: Validate OpenAPI schema generation.
- **Merge Goal**: Merge API.
- **Demo Goal**: Curl request returning inference.
- **Definition of Done**: Unit tests pass, OpenAPI schema locked.
- **Exit Criteria**: Frontend and SDK can begin building against endpoints.

## Sprint 4 (I4 & I5: SDK & Frontend)
- **Deliverables**: React components, SDK bindings.
- **Review Goal**: E2E data flow verification.
- **Merge Goal**: Merge consumer clients.
- **Demo Goal**: Live dashboard simulation.
- **Definition of Done**: Cypress/Playwright tests pass.
- **Exit Criteria**: Users can consume the API visually and programmatically.
