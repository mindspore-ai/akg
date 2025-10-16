#!/bin/bash

set -e

BACKEND_ENV="cpu"
TEST_TYPE="ut"
BASEPATH=$(cd $(dirname $0); pwd)
AKG_MLIR_BUILD_PATH=${BASEPATH}/../build
SYSTEM_TEST_DIR=${BASEPATH}/st
ST_TOOL=akg_benchmark

usage()
{
    echo "Usage:"
    echo "bash run_test.sh [-e cpu|gpu|ascend|all] [-t ut|st][-h]"
    echo ""
    echo "Options:"
    echo "    -h Print usage"
    echo "    -e Hardware environment: cpu, gpu, ascend or all"
    echo "    -t test type: ut, st or all (Default: ut)"
}

while getopts 'e:ht:' opt
do
    case "${opt}" in
        h)
            usage
            exit 0
            ;;
        e)
            if [[ "${OPTARG}" == "gpu" ]]; then
                BACKEND_ENV="gpu"
            elif [[ "${OPTARG}" == "ascend" ]]; then
                BACKEND_ENV="ascend"
            elif [[ "${OPTARG}" == "cpu" ]]; then
                BACKEND_ENV="cpu"
            elif [[ "${OPTARG}" == "all" ]]; then
                BACKEND_ENV="all"
            else
                echo "Unknown parameter ${OPTARG}!"
                usage
                exit 1
            fi
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

# run ut test cases
if [[ "X${TEST_TYPE}" = "Xut" ]] || [[ "X${TEST_TYPE}" = "Xall" ]]; then
    llvm-lit -sv ${AKG_MLIR_BUILD_PATH}/tests
fi

# run st test cases
if [[ "X${TEST_TYPE}" = "Xst" ]] || [[ "X${TEST_TYPE}" = "Xall" ]]; then
    # run st cpu test cases
    if [[ "X${BACKEND_ENV}" = "Xcpu" ]] || [[ "X${BACKEND_ENV}" = "Xall" ]]; then
        echo "Running the st cpu dynamic shape tests:"
        net=${SYSTEM_TEST_DIR}/dynamic_shape/
        if [[ $(${ST_TOOL} -e cpu -d ${net} -c 1 -ci 1 -t 16 | grep "dir test") == "dir test success" ]]; then
            echo "test ${net}: Success"
        else
            echo "test ${net}: Failed"
            exit 1
        fi

        echo "Running the st cpu tests:"
        network=$(ls ${SYSTEM_TEST_DIR}/cpu)
        for net in $network ;do
            echo "test ${net}"
            if [[ $(${ST_TOOL} -e cpu -d ${SYSTEM_TEST_DIR}/cpu/${net} -c 1 -ci 1 -t 16 | grep "dir test ") == "dir test success" ]]; then
                echo "test ${file}: Success"
            else
                echo "test ${file}: Failed"
                exit 1
            fi
        done
    fi
    # run st gpu test cases
    if [[ "X${BACKEND_ENV}" = "Xgpu" ]] || [[ "X${BACKEND_ENV}" = "Xall" ]]; then

        echo "Running the st gpu dynamic shape tests:"
        net=${SYSTEM_TEST_DIR}/gpu_dynamic_shape/
        if [[ $(${ST_TOOL} -e gpu -d ${net} -c 1 -ci 1 -t 8 | grep "dir test") == "dir test success" ]]; then
            echo "test ${net}: Success"
        else
            echo "test ${net}: Failed"
            exit 1
        fi
    fi
    # run st ascend test cases
    if [[ "X${BACKEND_ENV}" = "Xascend" ]] || [[ "X${BACKEND_ENV}" = "Xall" ]]; then
        echo "Running the st ascend tests:"
    fi

fi


