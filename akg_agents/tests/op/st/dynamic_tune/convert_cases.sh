#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
AKG_AGENTS_ROOT="${REPO_ROOT}/akg_agents"
TMP_ROOT="${AKG_AGENTS_ROOT}/workspace/.tmp"
POSTPROCESS_ROOT="${TMP_ROOT}/txa_postprocess"
CONVERTED_ROOT="${SCRIPT_DIR}/cases/converted"

cd "${REPO_ROOT}"

rm -rf "${TMP_ROOT}"
PYTHONPATH="${PWD}/akg_agents/python:${PWD}/akg_agents" \
    python -m pytest akg_agents/tests/op/st/dynamic_tune/test_dynamic_tune.py::test_convert -q "$@"

shopt -s nullglob
missing=0
for case_dir in "${CONVERTED_ROOT}"/*; do
    [[ -d "${case_dir}" ]] || continue
    case_name="$(basename "${case_dir}")"
    latest_dir=""
    latest_impl=""
    for impl_path in "${POSTPROCESS_ROOT}/${case_name}"-*/impl.py; do
        impl_dir="$(basename "$(dirname "${impl_path}")")"
        if [[ -z "${latest_dir}" || "${impl_dir}" > "${latest_dir}" ]]; then
            latest_dir="${impl_dir}"
            latest_impl="${impl_path}"
        fi
    done
    if [[ -z "${latest_impl}" ]]; then
        echo "missing postprocess output for ${case_name}" >&2
        missing=1
        continue
    fi
    cp -f "${latest_impl}" "${case_dir}/impl.py"
    echo "${latest_impl} -> ${case_dir}/impl.py"
done

exit "${missing}"
