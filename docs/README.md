# Documentation

This directory documents the public interface and operating model for
`safe-fs-ops`.

## Start Here

- [Public API](PUBLIC_API.md) is the main user-facing reference.
- [Recovery model](RECOVERY_MODEL.md) explains what the journal can recover and
  where manual intervention is expected.
- [Filesystem primitives](FILESYSTEM_PRIMITIVES.md) describes the lower-level
  helpers exposed from `safe_fs_ops.filesystem_ops`.
- [Development](DEVELOPMENT.md) covers local setup, validation, and publishing
  workflow notes.
- [Source review](SOURCE_REVIEW.md) records the current assessment of the source
  layout and public API shape.

## Audience

Most application code should use the top-level `safe_fs_ops` imports,
especially `SafeWorkspace`.

Callers that are building another coordination layer may also use:

- `safe_fs_ops.filesystem_ops`
- `safe_fs_ops.operation_journal`
- `safe_fs_ops.workspace_state`
- `safe_fs_ops.sqlite_store`
- `safe_fs_ops.persistence`

Those lower layers are public enough to build on, but they are more specialized
and may move faster before 1.0 than the top-level workspace API.
