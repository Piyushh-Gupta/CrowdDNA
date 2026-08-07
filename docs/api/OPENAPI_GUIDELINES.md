# OpenAPI Guidelines

The API will be fully documented using the OpenAPI 3.1.0 specification generated natively by FastAPI.

## Conventions
- **Tags**: Group endpoints strictly by domain (`Inference`, `System`, `Models`).
- **OperationIds**: Must be explicitly defined in snake_case (e.g., `submit_inference_job`).
- **Schema Naming**: Pydantic models must use PascalCase (e.g., `InferenceResponse`).
- **Descriptions**: Every field must have a `description` and `example` defined in the Pydantic `Field`.
