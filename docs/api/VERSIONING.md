# Versioning Strategy

## URL Versioning
The API will use URI routing for major versions.
Example: `/api/v1/inference`

## Semantic Versioning
- Major version increments (`v1` -> `v2`) indicate breaking changes.
- Minor/Patch additions will be added to the OpenAPI spec without URL increments.

## Breaking Change Policy
- Removing an endpoint.
- Changing a response field name or type.
- Making an optional request parameter required.
These will strictly require a new Major Version.
