#!/bin/bash

set -e

TEST_TYPE="ut"
UT_RUN_TYPE="all"
BASEPATH=$(cd "$(dirname $0)"; pwd)
MFUSION_BUILD_PATH=${BASEPATH}/../build

usage()
{
    echo "Usage:"
    echo "bash run_test.sh [-t ut|st|all] [-u lit|python|all] [-h]"
    echo ""
    echo "Options:"
    echo "    -h Print usage"
    echo "    -t test type: ut, st or all (Default: ut)"
    echo "    -u ut run type: lit, python or all (Default: all)"
}

while getopts 'ht:u:' opt
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
                echo "Unknown parameter for -t: ${OPTARG}!"
                usage
                exit 1
            fi
            ;;
        u)
            if [[ "${OPTARG}" == "lit" ]]; then
                UT_RUN_TYPE="lit"
            elif [[ "${OPTARG}" == "python" ]]; then
                UT_RUN_TYPE="python"
            elif [[ "${OPTARG}" == "all" ]]; then
                UT_RUN_TYPE="all"
            else
                echo "Unknown parameter for -u: ${OPTARG}!"
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

# run ut test cases (lit + python)
if [[ "X${TEST_TYPE}" = "Xut" ]] || [[ "X${TEST_TYPE}" = "Xall" ]]; then
    if [[ "X${UT_RUN_TYPE}" = "Xlit" ]] || [[ "X${UT_RUN_TYPE}" = "Xall" ]]; then
        bash "${BASEPATH}/ut/lit/runtest.sh"
    fi
    if [[ "X${UT_RUN_TYPE}" = "Xpython" ]] || [[ "X${UT_RUN_TYPE}" = "Xall" ]]; then
        bash "${BASEPATH}/ut/python/runtest.sh"
    fi
fi

# run st test cases (not supported yet)
if [[ "X${TEST_TYPE}" = "Xst" ]]; then
    echo "st tests are not supported in mfusion yet."
    exit 1
fi
