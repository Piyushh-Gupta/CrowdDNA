# API Change Policy

## Additive Changes
Adding new endpoints, new optional request fields, or new response fields are considered backwards-compatible additive changes. SDK clients must ignore unknown JSON keys.

## Deprecation Lifecycle
If an endpoint or field is to be deprecated:
1. It must be marked as `deprecated: true` in the OpenAPI spec.
2. A `Warning` HTTP header must be included in responses.
3. It must be supported for at least 6 months before removal.

## SDK Compatibility
The Python SDK must be strictly generated from or validated against the OpenAPI specification to prevent contract drift.
