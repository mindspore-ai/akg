#!/bin/bash

set -e

TEST_TYPE="ut"
BASEPATH=$(cd "$(dirname $0)"; pwd)
MFUSION_BUILD_PATH=${BASEPATH}/../build

usage()
{
    echo "Usage:"
    echo "bash run_test.sh [-t ut|st|all] [-h]"
    echo ""
    echo "Options:"
    echo "    -h Print usage"
    echo "    -t test type: ut, st or all (Default: ut)"
}

while getopts 'ht:' opt
do
    case "${opt}" in
        h)
            usage
            exit 0
            ;;
        t)
            if [[ "${OPTARG}" == "ut" ]]; then
                TEST_TYPE="ut"
            elif [[ "${OPTARG}" == "st" ]]; then
                TEST_TYPE="st"
            elif [[ "${OPTARG}" == "all" ]]; then
                TEST_TYPE="all"
            else
                echo "Unknown parameter ${OPTARG}!"
                usage
                exit 1
            fi
            ;;
        *)
            echo "Unknown option ${opt}!"
            usage
            exit 1
    esac
done

# run ut test cases (lit)
if [[ "X${TEST_TYPE}" = "Xut" ]] || [[ "X${TEST_TYPE}" = "Xall" ]]; then
    # check if tests directory exists
    TEST_PATH=${MFUSION_BUILD_PATH}/tests
    if [[ ! -d "${TEST_PATH}" ]]; then
        echo "Error: Tests directory is not built."
        echo "Please build with -t option (e.g., bash build.sh -t)"
        exit 1
    fi
    lit -sv ${TEST_PATH}
fi

# run st test cases (not supported yet)
if [[ "X${TEST_TYPE}" = "Xst" ]]; then
    echo "st tests are not supported in mfusion yet."
    exit 1
fi
