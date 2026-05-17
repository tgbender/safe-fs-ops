from __future__ import annotations


class JournalError(RuntimeError):
    pass


class BatchNotFoundError(JournalError):
    pass


class InvalidBatchPhaseTransitionError(JournalError):
    pass


class JournalLeaseMismatchError(JournalError):
    pass


class BatchIdempotencyMismatchError(JournalError):
    pass


class RecoveryAttemptMismatchError(JournalError):
    pass


class RecoveryContextError(JournalError):
    pass


class BatchStartConflictError(JournalError):
    pass
