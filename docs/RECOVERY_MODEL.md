# Recovery Model

`safe-fs-ops` does not make the filesystem transactional. It gives callers a
durable coordination and recovery model around operations that have explicit
proof.

## Core Rule

Recovery only acts from recorded evidence.

The journal records intent before mutation, then records observed checkpoints,
backup artifacts, failures, recovery plans, recovery attempts, and terminal
outcomes. If the package cannot prove that a recovery action is safe, it records
manual intervention required instead of guessing.

Releasing a workspace lease atomically revokes its secret token as well as
expiring it. A heartbeat delayed until after release cannot renew the old
authority, even if it sampled its clock before release. The next acquisition
continues to advance the fencing counter.

## What Gets Recorded

For a journaled mutation, the coordinator can record:

- operation intent and idempotency key
- owning lease and resource claim scope
- before checkpoints
- durable backup or capture artifacts
- filesystem mutation result
- after checkpoints
- failure observations
- recovery desired state
- recovery action intent and outcomes
- artifact cleanup debt and cleanup outcomes

The exact payloads are implementation details. The durable concept is that each
step leaves enough state for later readers to understand the safest next action.

## Rollback Modes

### `record-only`

`record-only` is the default. It records journal state and recovery-desired
status but does not automatically roll back the transaction body on failure.

Use this when:

- you want audit and explicit recovery control
- a workflow includes operations that are not automatically reversible
- you are invoking another tool and only want to record snapshots or backups

### `automatic`

`automatic` attempts to roll back supported transaction mutations when the
transaction body raises.

Automatic rollback is intentionally narrow. It is supported only when the
operation has durable rollback proof and the current filesystem state still
matches the expected state for recovery.

Supported high-level cases include:

- transaction-created file removal
- overwritten file restore from backup proof
- deleted file restore from backup proof
- transaction-created direct directory removal
- recursive mkdir rollback for transaction-owned created parents
- inverse no-replace rename recovery
- captured-directory restore from quarantine

Unsupported or unsafe cases are marked for recovery/manual intervention instead
of being forced.

## Pending Recovery

If a process crashes, loses a lease, or fails during cleanup, another workspace
instance can resume pending work:

```python
error = workspace.recover_pending_batches(run_id="run-123")
if error is not None:
    raise error
```

`run_id` is optional. Passing it scopes recovery to one logical run. Omitting it
allows the workspace to consider all eligible pending batches.

Recovery uses lease and recovery-action authority checks. Stale workers cannot
continue appending recovery actions after takeover.

## Legacy Directory Captures

Journals from older versions contain directory device/inode values but no
`capture_token`. Those values alone cannot prove that a quarantined directory
is still the original one. The workspace provides a supported recovery path:

```python
candidates = workspace.list_legacy_captures(run_id="run-123")
for candidate in candidates:
    print(candidate.capture_id, candidate.original_path,
          candidate.quarantine_path, candidate.status)
```

Inventory does not tag or move directories. After inspecting the contents and
confirming which particular directory belongs to the operation, select its
`LegacyCapture` object and call:

```python
workspace.recover_legacy_capture(
    selected_capture,
    confirm_ownership=True,
    reason="Operator inspected the quarantine and confirmed this capture",
)
```

This is an explicit ownership decision by the caller. It is not an automatic
upgrade based on a matching inode. The call checks the inspected directory's
identity and metadata again, holds an OS handle while adopting it, and records
the approver, reason, and intended token before writing the tag. A separate
recovery checkpoint records the installed proof. Old journal records remain
unchanged. Both checkpoints require the current lease and recovery attempt.

The operation resumes the selected capture's batch through the normal recovery
runner, including any other rollback actions already planned for that batch.
Other batches are not restored by this call. Multiple unconfirmed captures in
a batch require individual confirmations; previously completed adoptions remain
available if another capture still needs attention.

An existing destination, redirected path, changed identity, conflicting claim,
or missing confirmation is refused. If the directory changed since inspection,
inspect it again before confirming. The checks do not compare every file's
contents and cannot supply the caller's ownership decision. Inventory statuses
include `needs_confirmation`, `adopted`, `restored`, `identity_changed`, and
`unavailable`; the last two need investigation rather than automatic adoption.

After interruption before the proof checkpoint, repeat the explicit recovery
call. A persisted matching tag allows safe retry without treating its own
metadata update as tampering. Once proof is recorded, ordinary
`recover_pending_batches()` can resume, including after the directory was
restored but completion was not recorded. Adoption authorizes restoration and
sets the adopted artifact's cleanup policy to `retain`; it does not authorize
automatic deletion. No manual SQLite edits are needed.

## Artifact Cleanup

Some operations create durable artifacts:

- file backup content
- content-addressed tree backup objects
- captured directories in quarantine

Committed transactions may leave cleanup debt. Cleanup can be replayed:

```python
error = workspace.cleanup_outstanding_artifacts(run_id="run-123")
if error is not None:
    raise error
```

The default captured-directory cleanup policy is `"retain"` to keep the audit
trail and quarantine artifacts. Use `captured_directory_cleanup="automatic"`
when committed captured-directory artifacts should be cleaned after the journal
records a cleanup plan.

## Snapshots And Backups

`snapshot_bundle` records evidence without storing file content unless the hash
policy requests small-file hashes. It is useful for audit and risk
characterization, but it is not a restore artifact.

`backup_tree` stores named file content in a content-addressed store and records
symlink targets. It is a restore artifact for explicit paths, not a whole-tree
image.

`capture_directory` moves a directory into quarantine and records enough identity
information to restore the captured directory when recovery proof remains valid.

## Manual Intervention

Manual intervention is expected when:

- backup content is missing or tampered
- the current target state no longer matches expected recovery proof
- a path became unsafe, redirected, or unsupported
- another tool changed a destination after the journaled mutation
- the platform lacks the primitive needed for a safe recovery action
- a restore operation itself failed after partial progress

This behavior is deliberate. Silent overwrites are worse than a visible recovery
debt.

## Interaction With External Tools

The recommended pattern before invoking an untrusted or risky tool is:

1. Claim the resources that your own tool knows about.
2. Record `snapshot_bundle` evidence for broad context.
3. Record `backup_tree` artifacts for files you may need to restore.
4. Use `capture_directory` for directories you intend to remove or move.
5. Invoke the external tool outside the transaction or inside a surrounding
   workflow that understands the recovery boundary.
6. Use recorded artifacts for explicit restore workflows if needed.

`safe-fs-ops` cannot recover arbitrary unrecorded mutations made by another
process.
