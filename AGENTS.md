# CrowdDNA AI Development Guide
Whenever an instruction says "Follow the repository documentation", read and follow AGENTS.md, WORKFLOW.md, CODING_STANDARDS.md, PROJECT_STATUS.md, ROADMAP.md, ARCHITECTURE.md and DECISIONS.md before making architectural decisions.


## Default Development Workflow

Unless explicitly instructed otherwise, always:

1. Create a new feature branch from `develop`.
2. Implement only the requested module.
3. Follow ARCHITECTURE.md and DECISIONS.md.
4. Follow CODING_STANDARDS.md.
5. Add/update tests.
6. Run Ruff.
7. Run the complete test suite.
8. Commit with Conventional Commits.
9. Push the feature branch.
10. Return an implementation summary, assumptions, test results, commit message, and PR URL.
11. Never implement future phases or unrelated modules.

## Purpose

This repository uses AI-assisted development. Follow these rules for every implementation task.

---

## General Rules

- Reuse the existing architecture.
- Modify only the requested component.
- Avoid unrelated changes.
- Do not rewrite working code.
- Preserve backward compatibility unless explicitly instructed otherwise.
- Keep the implementation modular and production-ready.

---

## Coding Standards

- Use Python type hints everywhere.
- Use concise Google-style docstrings.
- Follow PEP 8.
- Prefer readability over clever code.
- Avoid duplicated logic.
- Keep functions focused on a single responsibility.
- Avoid global mutable state.
- Use deterministic behaviour whenever randomness is involved.

---

## Code Quality

Before considering a task complete:

- Ruff must pass.
- All tests must pass.
- Add or update smoke tests when functionality changes.
- Remove unused imports.
- Remove dead code.
- Do not leave TODO or FIXME placeholders.

---

## Architecture Rules

- Respect the existing project structure.
- Do not introduce unnecessary abstractions.
- Do not modify unrelated modules.
- Keep responsibilities separated.

Example:

```
ScenarioFactory     → builds scenarios
SimulationRunner    → runs simulations
AutoLabeler         → labels trajectories
DatasetSerializer   → writes datasets
```

Each module should have one responsibility.

---

## Documentation

Every new public class or function should contain:

- A concise Google-style docstring.
- Parameter descriptions when useful.
- Return value description when useful.

Avoid excessive comments.

Code should be self-explanatory.

---

## If Uncertain

Never invent architecture.

Follow the SRD and existing implementation.