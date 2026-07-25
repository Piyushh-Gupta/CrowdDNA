# Code Review Guide

## Review Checklists

### Architecture
- [ ] Does this align with the frozen architecture?
- [ ] Are abstraction boundaries respected?

### Security
- [ ] Are secrets exposed?
- [ ] Have security validations been added?

### Performance
- [ ] Are there obvious bottlenecks?
- [ ] Is memory managed correctly?

### Testing
- [ ] Are there unit tests?
- [ ] Do tests cover edge cases?

### Documentation
- [ ] Are new public APIs documented?
- [ ] Are docstrings present?

### Accessibility
- [ ] (Frontend) Are ARIA labels used?
- [ ] (Frontend) Is color contrast sufficient?

## SLAs
- Reviews should be completed within 24 hours of PR creation.

## Blocking vs Non-Blocking
- **Blocking**: Architectural violations, security risks, broken tests, missing documentation.
- **Non-Blocking**: Minor formatting issues, non-critical optimizations (leave as follow-up).
