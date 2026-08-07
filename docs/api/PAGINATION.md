# Pagination Strategy

For list endpoints (e.g., `GET /jobs` or `GET /models`), cursor-based pagination will be used.

## Query Parameters
- `limit` (Integer): Maximum items to return. Default: 50. Max: 100.
- `cursor` (String): Opaque string pointing to the next page.

## Response Format
```json
{
  "data": [...],
  "next_cursor": "c3RhcnRfdGltZTo..."
}
```
