# Source Review

This review summarizes the current source and public interface shape after
reading `src/safe_fs_ops`.

## Overall Assessment

The package has a coherent architecture for a pre-1.0 filesystem safety library.
The best public entry point is clearly `SafeWorkspace`; the lower layers are
separable and useful for advanced callers, but they should be treated as builder
APIs rather than the normal application surface.

The main design is sound: durable SQLite state owns cross-process authority,
Python locks protect in-process coordinator state, filesystem mutations are
typed, and recovery relies on explicit proof instead of arbitrary command
rollback.

## Public API Shape

The top-level `safe_fs_ops` namespace is clean and small:

- resources: `FileResource`, `DirectoryResource`, `TreeResource`,
  `ResourceHandle`, `ResourceSet`
- coordination: `SafeWorkspace`, `SafeTransaction`, `SafeOperation`,
  `SafePhase`
- errors: `SafeWorkspaceError`, `SafeWorkspaceBusyError`,
  `ResourceNotClaimedError`
- durability: `DurabilityMode`

That is the right primary API for application code.

The package-level exports under `filesystem_ops`, `operation_journal`,
`workspace_state`, `sqlite_store`, and `persistence` are much wider. They are
valuable, but they expose more implementation vocabulary. Documentation should
steer most users to `SafeWorkspace` and reserve those modules for framework
builders.

## Strengths

- `SafeWorkspace.open(...)` is explicit and dependency-injection friendly.
- Resource handles make conflict keys visible and testable.
- Transactions claim resources before mutation and reject unclaimed resources.
- Recovery concepts are separated: leases, claims, batches, checkpoints,
  recovery actions, and artifact cleanup are distinct.
- `SafeTransaction` and `SafePhase` expose the same core mutation vocabulary.
- Lower-level filesystem operations are conservative about symlinks, reparse
  points, mount points, parent-chain changes, and unsupported platform
  primitives.
- Content-addressed artifacts give restore operations something concrete to
  verify.
- Tree backups are deliberately explicit instead of pretending to be cheap,
  complete directory images.
- SQLAlchemy-backed persistence exists behind typed store APIs rather than
  exposing raw SQL as the public interface.

## Sharp Edges

- The project is alpha. Public package exports are broad, especially
  `operation_journal`, so the compatibility boundary needs to be documented.
- `restore_tree_backup` is journaled but intentionally rejected inside
  `rollback="automatic"` transactions. Callers need to use record-only recovery
  workflows for restores.
- `backup_tree` captures explicit files and symlinks only. It does not capture
  empty directories, directory permissions, or a complete recursive tree.
- `snapshot_bundle` is evidence, not a restore artifact.
- Automatic rollback is proof-dependent. It may leave manual-intervention debt
  when another process changes the target after proof capture.
- Phased operations currently support `record-only` rollback only.
- Custom filesystem backends can be injected, but automatic file rollback should
  remain opt-in unless the backend follows the required proof semantics.
- `persistence.__init__` describes itself as internal while exporting public
  helpers. The practical stance should be "advanced public, not primary API"
  until 1.0 clarifies the boundary.

## Fit For `wt` And `prescribe`

The package fits the risky parts of those workflows when used as a proof and
recovery layer:

- use `snapshot_bundle` before invoking another tool for audit evidence
- use `backup_tree` for files that may need restoration
- use `capture_directory` before moving or tearing down `.git`-containing
  directories
- use `recover_pending_batches` and `cleanup_outstanding_artifacts` on startup
  or after interrupted runs

It should not absorb arbitrary domain logic from those projects. Domain-specific
decisions should stay in the caller; `safe-fs-ops` should provide typed,
recoverable primitives.

## Remaining Risks

- Native filesystem behavior varies across Windows, Linux, WSL, macOS, network
  filesystems, and antivirus/indexing interference. The library should continue
  to be validated on real target platforms.
- The filesystem cannot provide full transaction isolation. The package reduces
  risk; it cannot make malicious or incompetent external actors harmless.
- Artifact retention and cleanup policies need operational discipline. Retained
  backups are useful for recovery but can consume disk space.
- Broad lower-level exports may need a compatibility pass before a non-alpha
  release.

## Recommendation

Use the package as an alpha dependency behind a caller-owned recovery workflow.
The top-level workspace API is clean enough to integrate. Publish as pre-1.0
only, keep examples focused on explicit proof, and avoid marketing it as a
general-purpose transactional filesystem.
