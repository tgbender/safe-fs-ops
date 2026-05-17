from __future__ import annotations

from safe_fs_ops.resources import DirectoryResource, FileResource, ResourceHandle, ResourceSet


def require_file_resource(
    *,
    resources: ResourceSet,
    resource: FileResource,
    resource_not_claimed_error: type[Exception],
) -> None:
    if not isinstance(resource, FileResource):
        raise TypeError("transaction file operations require a FileResource")
    require_claimed_resource(
        resources=resources,
        resource=resource,
        resource_not_claimed_error=resource_not_claimed_error,
    )


def require_directory_resource(
    *,
    resources: ResourceSet,
    resource: DirectoryResource,
    resource_not_claimed_error: type[Exception],
) -> None:
    if not isinstance(resource, DirectoryResource):
        raise TypeError("transaction directory operations require a DirectoryResource")
    require_claimed_resource(
        resources=resources,
        resource=resource,
        resource_not_claimed_error=resource_not_claimed_error,
    )


def require_claimed_resource(
    *,
    resources: ResourceSet,
    resource: ResourceHandle,
    resource_not_claimed_error: type[Exception],
) -> None:
    if resources.contains_resource(resource):
        return
    if resources.contains_resource_key(resource.resource_key):
        raise resource_not_claimed_error(
            f"resource {resource.resource_key!r} was not claimed by this transaction; "
            "use the claimed ResourceSet member from tx.r"
        )
    raise resource_not_claimed_error(f"resource {resource.resource_key!r} was not claimed by this transaction")


def next_idempotency_key(
    *,
    name: str,
    run_id: str,
    operation_index: int,
    operation: str,
    resource: ResourceHandle,
) -> tuple[int, str]:
    next_index = operation_index + 1
    label = resource.label or resource.resource_key
    return next_index, f"{name}:{run_id}:{next_index}:{operation}:{label}"
