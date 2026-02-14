from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-large",
        action="store_true",
        default=False,
        help="Run tests marked 'large' (scale-representative, e.g. N=256).",
    )
    parser.addoption(
        "--run-stress",
        action="store_true",
        default=False,
        help="Run tests marked 'stress' (very large, e.g. N=512).",
    )


def pytest_collection_modifyitems(config, items):
    run_large = config.getoption("--run-large")
    run_stress = config.getoption("--run-stress")

    skip_large = pytest.mark.skip(reason="need --run-large to run")
    skip_stress = pytest.mark.skip(reason="need --run-stress to run")

    for item in items:
        if "stress" in item.keywords and not run_stress:
            item.add_marker(skip_stress)
        elif "large" in item.keywords and not run_large:
            item.add_marker(skip_large)
