# Branching Strategy

Our strategy is based on strict PR-driven development.

```mermaid
graph LR
    main --> develop
    develop --> feature[feature/feature-name]
    feature --> develop
```

## Rules
- **Never commit directly to `main`.**
- **Never commit directly to `develop`.**
- Everything must go through a Pull Request.
- Feature branches should be deleted after merging.
