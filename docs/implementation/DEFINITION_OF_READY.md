# Definition of Ready (DoR)

Before any task or subtask is moved into the "In Progress" column, it must meet the following minimum requirements:

1. **Requirements Defined**: The issue clearly states what needs to be implemented.
2. **Dependencies Resolved**: All blocking upstream tasks (e.g., API schemas) are merged into `develop`.
3. **Acceptance Criteria Written**: Explicit Boolean conditions defining success are documented in the issue.
4. **Owner Assigned**: Exactly one developer is assigned as the primary implementer.
5. **Reviewer Assigned**: Exactly one developer is assigned as the primary reviewer.
6. **Estimate Added**: Effort estimation in hours or days is provided.
7. **Test Strategy Defined**: The specific validation (e.g., unit test, Playwright E2E) is listed.
