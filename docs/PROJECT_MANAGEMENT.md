# GitHub Project Management

## Board Columns

1. **Backlog**: All created issues waiting to be prioritized.
2. **Ready**: Issues that are well-defined and ready for work. (Entry: fully spec'd. Exit: assigned to a sprint.)
3. **Assigned**: Task assigned to a specific developer. (Entry: sprint starts. Exit: development begins.)
4. **In Progress**: Developer is actively working on it. (Entry: coding begins. Exit: PR created.)
5. **Code Review**: PR is open and awaiting review. (Entry: PR created. Exit: PR approved.)
6. **Testing**: Merged to `develop` and awaiting QA/validation.
7. **Blocked**: Development is halted due to a dependency or issue.
8. **Done**: Fully verified and complete.

## Issue Types
- Epic, Feature, Task, Bug, Spike, Research, Technical Debt, Chore, Documentation.

## Labels
- **Priority**: `P0`, `P1`, `P2`, `P3`
- **Area**: `backend`, `frontend`, `ml`, `sdk`, `deployment`, `security`, `documentation`, `infrastructure`
- **Status**: `blocked`, `ready`, `review`, `testing`
- **Difficulty**: `good first issue`, `beginner`, `intermediate`, `advanced`

## Milestones
- Architecture Complete, Implementation I1, Implementation I2, ..., Beta, Release Candidate, v1.0

## Sprint Planning
- Weekly Planning on Mondays.
- Task Assignment by Technical Lead.
- Daily async updates.
- Review Cycle driven by SLAs.
- Weekly Retrospective on Fridays.
