#!/usr/bin/env bash
set -euo pipefail

MODE="dev"
REPO="testpypi"
PUBLISH="0"
SET_VERSION=""

usage() {
  cat <<'EOF'
用法:
  bash akg_agents/scripts/publish_akg_cli.sh --version VERSION [--mode dev|post|patch] [--repo testpypi|pypi] [--publish]

说明:
  - version 为必填，脚本不再自动递增版本号
  - 默认 mode=dev，repo=testpypi
  - 默认仅更新版本号（不构建、不上传）
  - --publish 才会更新版本号并执行构建与上传
  - mode 仅用于标识/记录（不影响版本号）
  - 上传需要 TWINE_PASSWORD 环境变量
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --publish)
      PUBLISH="1"
      shift
      ;;
    --version)
      SET_VERSION="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AIKG_DIR="${ROOT_DIR}"
CLI_DIR="${ROOT_DIR}/akg-cli"

if [[ ! -f "${AIKG_DIR}/version.txt" ]]; then
  echo "Missing ${AIKG_DIR}/version.txt" >&2
  exit 1
fi
if [[ ! -d "${CLI_DIR}" ]]; then
  echo "Missing ${CLI_DIR} (akg-cli wrapper)" >&2
  exit 1
fi

CURRENT_VERSION="$(cat "${AIKG_DIR}/version.txt" | tr -d '[:space:]')"
if [[ -z "${SET_VERSION}" ]]; then
  echo "--version is required" >&2
  exit 1
fi
NEXT_VERSION="${SET_VERSION}"

if ! V="${NEXT_VERSION}" python - <<'PY'
import os, re, sys
v = os.environ["V"]
pattern = r"^(\d+\.\d+\.\d+)(?:\.post(\d+))?(?:\.dev(\d+))?$"
if not re.match(pattern, v):
    print(f"Unsupported version format: {v}", file=sys.stderr)
    sys.exit(1)
PY
then
  exit 1
fi

if [[ "${PUBLISH}" = "0" ]]; then
  echo "Current: ${CURRENT_VERSION}"
  echo "Next:    ${NEXT_VERSION}"
  echo "==> No changes applied (use --publish to update/build/upload)"
  exit 0
fi

echo "Version bump: ${CURRENT_VERSION} -> ${NEXT_VERSION}"

echo "${NEXT_VERSION}" > "${AIKG_DIR}/version.txt"
echo "${NEXT_VERSION}" > "${CLI_DIR}/version.txt"

if [[ -z "${TWINE_PASSWORD:-}" ]]; then
  echo "TWINE_PASSWORD is required for upload" >&2
  exit 1
fi

TWINE_USERNAME="${TWINE_USERNAME:-__token__}"

python -m pip install -U build twine >/dev/null

echo "==> Build akg_agents"
(cd "${AIKG_DIR}" && python -m build --no-isolation)

echo "==> Build akg-cli"
(cd "${CLI_DIR}" && python -m build --no-isolation)

echo "==> Upload akg_agents"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  TWINE_USERNAME="${TWINE_USERNAME}" TWINE_PASSWORD="${TWINE_PASSWORD}" \
  twine upload -r "${REPO}" --skip-existing "${AIKG_DIR}/dist/akg_agents-${NEXT_VERSION}"*

echo "==> Upload akg-cli"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  TWINE_USERNAME="${TWINE_USERNAME}" TWINE_PASSWORD="${TWINE_PASSWORD}" \
  twine upload -r "${REPO}" --skip-existing "${CLI_DIR}/dist/akg_cli-${NEXT_VERSION}"*
