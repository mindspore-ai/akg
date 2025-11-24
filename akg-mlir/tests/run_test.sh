#!/bin/bash

set -e

TEST_TYPE="ut"
BASEPATH=$(cd "$(dirname $0)"; pwd)
AKG_MLIR_BUILD_PATH=${BASEPATH}/../build

usage()
{
    echo "Usage:"
    echo "bash run_test.sh [-t ut|st][-h]"
    echo ""
    echo "Options:"
    echo "    -h Print usage"
    echo "    -m pytest mark"
    echo "    -t test type: ut, st or all (Default: ut)"
}

while getopts 'hm:t:' opt
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
        m)
            TEST_MARK=${OPTARG}
            ;;
        *)
            echo "Unknown option ${opt}!"
            usage
            exit 1
    esac
done

# run ut test cases
if [[ "X${TEST_TYPE}" = "Xut" ]] || [[ "X${TEST_TYPE}" = "Xall" ]]; then
    llvm-lit -sv ${AKG_MLIR_BUILD_PATH}/tests
fi

# run st test cases
if [[ "X${TEST_TYPE}" = "Xst" ]] || [[ "X${TEST_TYPE}" = "Xall" ]]; then
  if [[ -z ${TEST_MARK} ]]; then
    echo "run st must set pytest mark"
    usage
    exit 1
  fi
  cd ${BASEPATH}/st
  pytest -m ${TEST_MARK}
  cd -
fi


