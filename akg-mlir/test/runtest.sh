#!/bin/bash

set -e

if [ $# -lt 1 ]; then
echo "Must specify backend, cpu/CPU or gpu/GPU."
exit 1;
fi

BACKEND_ENV=$1
BASEPATH=$(cd $(dirname $0); pwd)
AKG_MLIR_BUILD_PATH=${BASEPATH}/../../build/akg-mlir
SYSTEM_TEST_DIR=${BASEPATH}/st
TOOL_PATH=${BASEPATH}/../python/akg_v2/exec_tools
ST_TOOL=${TOOL_PATH}/py_benchmark.py

export PATH=${AKG_MLIR_BUILD_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${AKG_MLIR_BUILD_PATH}/lib:${LD_LIBRARY_PATH}

# 1. run ut test cases
cmake --build ${AKG_MLIR_BUILD_PATH}/test/ --target check-akg-mlir

# 2. run st test cases
if [[ "X${BACKEND_ENV}" = "XCPU" ]] || [[ "X${BACKEND_ENV}" = "Xcpu" ]]; then
    echo "Running the st cpu dynamic shape tests:"
    net=${SYSTEM_TEST_DIR}/dynamic_shape/
    rm -rf ${TOOL_PATH}/akg_kernel_meta
    rm -rf ${TOOL_PATH}/mlir_files
    rm -rf ${TOOL_PATH}/tmp_files
    if [[ $(python ${ST_TOOL} -e cpu -d ${net} -c 1 -ci 1 -t 16 | grep "dir test") == "dir test success" ]]; then
        echo "test ${net}: Success"
    else
        echo "test ${net}: Failed"
        exit 1
    fi

    echo "Running the st cpu tests:"
    network=$(ls ${SYSTEM_TEST_DIR}/cpu)
    for net in $network ;do
	echo "test ${net}"
        rm -rf ${TOOL_PATH}/akg_kernel_meta
        rm -rf ${TOOL_PATH}/mlir_files
        rm -rf ${TOOL_PATH}/tmp_files
        if [[ $(python ${ST_TOOL} -e cpu -d ${SYSTEM_TEST_DIR}/cpu/${net} -c 1 -ci 1 -t 16 | grep "dir test ") == "dir test success" ]]; then
            echo "test ${file}: Success"
        else
            echo "test ${file}: Failed"
            exit 1
        fi
    done
elif [[ "X${BACKEND_ENV}" = "XGPU" ]] || [[ "X${BACKEND_ENV}" = "Xgpu" ]]; then

    echo "Running the st gpu dynamic shape tests:"
    cd ${TOOL_PATH}

    net=${SYSTEM_TEST_DIR}/gpu_dynamic_shape/
    rm -rf ${TOOL_PATH}/akg_kernel_meta
    rm -rf ${TOOL_PATH}/mlir_files
    rm -rf ${TOOL_PATH}/tmp_files
    if [[ $(python ${ST_TOOL} -e gpu -d ${net} -c 1 -ci 1 -t 8 | grep "dir test") == "dir test success" ]]; then
        echo "test ${net}: Success"
    else
        echo "test ${net}: Failed"
        exit 1
    fi

    echo "Running the st gpu tests:"
    network=$(ls ${SYSTEM_TEST_DIR}/gpu)
    for net in $network ;do
        echo "test ${net}"
        cd ${TOOL_PATH}
        rm -rf ${TOOL_PATH}/akg_kernel_meta
        rm -rf ${TOOL_PATH}/mlir_files
        rm -rf ${TOOL_PATH}/tmp_files
        if [[ $(python ${ST_TOOL} -e gpu -d ${SYSTEM_TEST_DIR}/gpu/${net} -c 1 -ci 1 -t 16 | grep "dir test ") == "dir test success" ]]; then
            echo "test ${net}: Success"
        else
            echo "test ${net}: Failed"
            exit 1
        fi
    done

    cd -
else
    echo "Only CPU and GPU supported"
fi
