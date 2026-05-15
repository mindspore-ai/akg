#!/bin/bash

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
MFUSION_ROOT_PATH=${BASEPATH}/../../..

TEST_PATH=${MFUSION_ROOT_PATH}/build/tests
if [[ ! -d "${TEST_PATH}" ]]; then
    echo "Error: Tests directory is not built."
    echo "Please build with -t option (e.g., bash build.sh -t)"
    exit 1
fi

lit -sv ${TEST_PATH}
