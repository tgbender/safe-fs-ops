from __future__ import annotations

import os
import sys

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--stress-locks",
        action="store_true",
        default=False,
        help="Run opt-in multi-process lock contention stress tests.",
    )
    parser.addoption(
        "--slow-recovery",
        action="store_true",
        default=False,
        help="Run slow recovery and automatic rollback integration tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "stress_lock: opt-in multi-process lock contention tests; run with --stress-locks.",
    )
    config.addinivalue_line(
        "markers",
        "slow_recovery: slow recovery and automatic rollback integration tests; run with --slow-recovery.",
    )
    config.addinivalue_line(
        "markers",
        "platform_windows: tests that require native Windows runtime support.",
    )
    config.addinivalue_line(
        "markers",
        "platform_posix: tests that require POSIX runtime support.",
    )
    config.addinivalue_line(
        "markers",
        "platform_macos: tests that require native macOS runtime support.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--stress-locks"):
        skip_stress_locks = pytest.mark.skip(reason="run with --stress-locks")
        for item in items:
            if item.get_closest_marker("stress_lock") is not None:
                item.add_marker(skip_stress_locks)
    if not config.getoption("--slow-recovery"):
        skip_slow_recovery = pytest.mark.skip(reason="run with --slow-recovery")
        for item in items:
            if item.get_closest_marker("slow_recovery") is not None:
                item.add_marker(skip_slow_recovery)
    platform_markers = (
        (
            "platform_windows",
            os.name != "nt",
            "requires native Windows runtime support",
        ),
        (
            "platform_posix",
            os.name == "nt",
            "requires POSIX runtime support",
        ),
        (
            "platform_macos",
            sys.platform != "darwin",
            "requires native macOS runtime support",
        ),
    )
    for marker_name, should_skip, reason in platform_markers:
        if not should_skip:
            continue
        skip_platform = pytest.mark.skip(reason=reason)
        for item in items:
            if item.get_closest_marker(marker_name) is not None:
                item.add_marker(skip_platform)
