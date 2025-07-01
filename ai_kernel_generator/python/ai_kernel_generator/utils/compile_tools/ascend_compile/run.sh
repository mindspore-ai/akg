# #!/bin/bash
# CURRENT_DIR=$(
#     cd $(dirname ${BASH_SOURCE:-$0})
#     pwd
# )

# BUILD_TYPE="Debug"
# INSTALL_PREFIX="${CURRENT_DIR}/output"

# SHORT=o:,r:,v:,i:,b:,p:,
# LONG=op-name:,run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
# OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
# eval set -- "$OPTS"

# while :; do
#     case "$1" in
#     -o | --op-name)
#         OP_NAME="$2"
#         shift 2
#         ;;
#     -r | --run-mode)
#         RUN_MODE="$2"
#         shift 2
#         ;;
#     -v | --soc-version)
#         SOC_VERSION="$2"
#         shift 2
#         ;;
#     -i | --install-path)
#         ASCEND_INSTALL_PATH="$2"
#         shift 2
#         ;;
#     -b | --build-type)
#         BUILD_TYPE="$2"
#         shift 2
#         ;;
#     -p | --install-prefix)
#         INSTALL_PREFIX="$2"
#         shift 2
#         ;;
#     --)
#         shift
#         break
#         ;;
#     *)
#         echo "[ERROR] Unexpected option: $1"
#         break
#         ;;
#     esac
# done

# RUN_MODE_LIST="cpu sim npu"
# if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
#     echo "ERROR: RUN_MODE error, This sample only support specify cpu, sim or npu!"
#     exit -1
# fi

# VERSION_LIST="Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
# if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
#     echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
#     exit -1
# fi

# if [ -n "$ASCEND_INSTALL_PATH" ]; then
#     _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
# elif [ -n "$ASCEND_HOME_PATH" ]; then
#     _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
# else
#     _ASCEND_INSTALL_PATH=/usr/local/Ascend/latest
# fi

# export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
# export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
# source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
# if [ "${RUN_MODE}" = "sim" ]; then
#     # in case of running op in simulator, use stub .so instead
#     export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
#     if [ ! $CAMODEL_LOG_PATH ]; then
#         export CAMODEL_LOG_PATH=$(pwd)/sim_log
#     fi
#     if [ -d "$CAMODEL_LOG_PATH" ]; then
#         rm -rf $CAMODEL_LOG_PATH
#     fi
#     mkdir -p $CAMODEL_LOG_PATH
# elif [ "${RUN_MODE}" = "cpu" ]; then
#     export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
# fi

# set -e
# rm -rf build output
# mkdir -p build
# cmake -B build \
#     -DOP_NAME=${OP_NAME} \
#     -DRUN_MODE=${RUN_MODE} \
#     -DSOC_VERSION=${SOC_VERSION} \
#     -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
#     -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
#     -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
# cmake --build build -j
# cmake --install build

# rm -f ascendc_aikg
# cp ./output/bin/ascendc_aikg ./
# rm -rf data/*
# mkdir -p data/input data/output
# python3 test_${OP_NAME}.py
# (
#     export LD_LIBRARY_PATH=$(pwd)/output/lib:$(pwd)/output/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
#     echo "LD_LIBRARY_PATH:"
#     echo ${LD_LIBRARY_PATH}
#     if [[ "$RUN_WITH_TOOLCHAIN" -eq 1 ]]; then
#         if [ "${RUN_MODE}" = "npu" ]; then
#             msprof op --application=./ascendc_aikg
#         elif [ "${RUN_MODE}" = "sim" ]; then
#             msprof op simulator --application=./ascendc_aikg
#         elif [ "${RUN_MODE}" = "cpu" ]; then
#             ./ascendc_aikg
#         fi
#     else
#         ./ascendc_aikg
#     fi
# )
# md5sum data/output/*.bin
# # bash run.sh -r npu -v Ascend910B4
