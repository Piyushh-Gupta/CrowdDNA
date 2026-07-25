# Style Guide

## Markdown Formatting
- Use **GitHub Flavored Markdown (GFM)**.
- **Heading Hierarchy**: Exactly one `<h1>` (`#`) per document. Descend logically (`##`, `###`).
- **Code Blocks**: Always provide a language hint (e.g., ` ```python `).
- **Tables**: Use standard pipe syntax, aligning headers.
- **Mermaid Diagrams**: Prefer `mermaid` fenced blocks for architectural visuals.

## Terminology & Naming
- Use precise nomenclature defined in [GLOSSARY.md](GLOSSARY.md).
- File names must be `UPPER_SNAKE_CASE.md` for root documentation files.

## Cross References
- Always use relative markdown links (e.g., `[Link](../README.md)`). Avoid absolute paths or HTTP URLs for internal documentation.
