# CrowdDNA Coding Standards

## Naming

**Classes** — PascalCase

**Functions** — snake_case

**Variables** — snake_case

**Constants** — UPPER_CASE

**Private methods** — prefix with `_`

---

## Imports

Order imports as follows:

1. Standard library
2. Third-party
3. Local modules

Remove unused imports.

---

## Functions

Prefer functions that are:

- Small
- Deterministic
- Testable

Avoid functions with multiple responsibilities.

---

## Classes

Each class should solve one problem.

Avoid God classes.

---

## Errors

- Raise meaningful exceptions.
- Wrap third-party exceptions when appropriate.
- Do not silently ignore errors.

---

## Testing

Every implemented feature should include:

- Smoke tests.
- Edge case tests when appropriate.

Existing tests must continue passing.

---

## Performance

Prioritize in this order:

```
Correctness
  ↓
Readability
  ↓
Maintainability
  ↓
Performance
```

Optimize only after correctness is established.

---

## JSON

Generated JSON should be:

- Deterministic.
- Human-readable.
- Consistently formatted.

---

## Randomness

Never use global random state.

Prefer:

```python
np.random.default_rng(seed)
```

instead of:

```python
np.random.seed()
```

---

## Logging

- Avoid excessive logging.
- Only log meaningful events.

---

## Documentation

- Public APIs must be documented.
- Private helper methods should be documented only when the logic is non-obvious.