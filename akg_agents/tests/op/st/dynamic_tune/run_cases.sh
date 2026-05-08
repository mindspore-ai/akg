#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

cd "${REPO_ROOT}"

PYTHONPATH="${PWD}/akg_agents/python:${PWD}/akg_agents" \
    python -m pytest akg_agents/tests/op/st/dynamic_tune/test_dynamic_tune.py::test_run -q "$@"
