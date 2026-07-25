# Contributing to CrowdDNA

## Engineering Workflow

1. **Feature Branch**: Create a branch off `develop` (`feat/`, `fix/`, `chore/`).
2. **Implementation**: Write code adhering to [CODING_STANDARDS.md](CODING_STANDARDS.md).
3. **Architecture Review**: Submit an ADR if making systemic changes ([DECISION_LOG.md](DECISION_LOG.md)).
4. **Engineering Audit**: Self-audit for concurrency, determinism, and performance.
5. **Validation**: Run `python -m pytest tests/` and `ruff check .`.
6. **Commit**: Use Conventional Commits.
7. **Pull Request**: Open a PR targeting `develop` using the [Pull Request Template](../.github/PULL_REQUEST_TEMPLATE.md).
8. **Code Review**: Requires at least one maintainer approval.
9. **Merge**: Squashed or Rebased into `develop`.
10. **Release Promotion**: Merged into `main` periodically via the [RELEASE_PROCESS.md](RELEASE_PROCESS.md).
