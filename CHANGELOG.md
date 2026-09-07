# Changelog

## 0.1.1

- Persist directory capture tokens to reject recycled filesystem identities.
- Support native macOS capture tokens and explicit recovery of legacy captures.
- Correct Windows UTF-16 rename buffers, reject embedded NULs, and declare native
  syscall argument and return types.
- Revoke released lease authority so delayed heartbeats cannot reactivate it.
- Flush capture tokens when retrying an existing Windows metadata stream.
- Expand native failure, ABI, lease race, and crash recovery regression coverage.
- Create release tags at the commit used by the publishing workflow.

## 0.1.0

- Initial release of filesystem operations, journaling, and workspace recovery.
