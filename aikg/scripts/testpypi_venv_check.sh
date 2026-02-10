#!/usr/bin/env bash
set -euo pipefail

PKG_NAME="${1:-ai-kernel-generator}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_VERSION="$(cat "${SCRIPT_DIR}/../version.txt" 2>/dev/null || echo "0.1.0")"
PKG_VERSION="${2:-${DEFAULT_VERSION}}"
TESTPYPI_INDEX="${TESTPYPI_INDEX:-https://test.pypi.org/simple}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"
CLEANUP="${CLEANUP:-0}"
NO_CACHE="${NO_CACHE:-0}"

VENV_DIR="$(mktemp -d /tmp/akg_cli_venv_testpypi_XXXXXX)"
echo "==> Creating venv: ${VENV_DIR}"
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Installing ${PKG_NAME}==${PKG_VERSION} from TestPyPI (single pip install)"
python -m pip install -U pip >/dev/null
PIP_FLAGS=()
if [ "${NO_CACHE}" = "1" ]; then
  PIP_FLAGS+=(--no-cache-dir)
fi
python -m pip install -i "${TESTPYPI_INDEX}" --extra-index-url "${PYPI_INDEX}" "${PKG_NAME}==${PKG_VERSION}" \
        --trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org \
        --trusted-host pypi.org --trusted-host files.pythonhosted.org \
        "${PIP_FLAGS[@]}"

echo "==> Running akg_cli --help"
akg_cli --help | head -n 40

echo "==> Done"
echo "venv: ${VENV_DIR}"
if [ "${CLEANUP}" = "1" ]; then
  rm -rf "${VENV_DIR}"
  echo "==> Cleanup complete"
fi
