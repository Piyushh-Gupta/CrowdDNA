# CrowdDNA Git Workflow

## Branch Strategy

Never commit directly to:

- `main`
- `develop`

Always create a feature branch:

```
feature/simulation-runner
feature/auto-labeler
feature/dataset-serializer
```

---

## Development Flow

```
develop
  ↓
feature/<feature>
  ↓
Implement
  ↓
Run Ruff
  ↓
Run Tests
  ↓
Add/Update Smoke Tests
  ↓
Commit
  ↓
Push
  ↓
Open Pull Request → develop
  ↓
Wait for Review
  ↓
Merge
  ↓
Delete Feature Branch
```

---

## Pull Requests

Every pull request should contain:

- A meaningful title.
- A summary.
- Implementation details.
- Validation performed.
- Assumptions (if any).

---

## Merge Rules

Merge only after:

- Review is completed.
- Ruff passes.
- Tests pass.
- CI passes.

Never merge a failing CI.

---

## Main Branch

`main` represents stable milestones only.

Merge `develop` into `main` only after an entire milestone or phase has been completed.