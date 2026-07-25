# Release Process

CrowdDNA separates release engineering from deployment. 

## Lifecycle

1. **Draft**: Create a semantic version bump via `VersionManager`.
2. **Release Candidate (RC)**: Promote to RC. Artifacts (wheels, sdists) are generated.
3. **Validation**: Artifacts undergo SBOM extraction, checksum signing (SHA256), and license validation.
4. **Freeze**: RC is frozen and cannot be mutated.
5. **Publish**: Artifacts are published to GitHub, PyPI, and Docker registries.

See the architecture in the [Phase 28 Release Framework](RELEASE_ARCHITECTURE.md).
