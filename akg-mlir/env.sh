#!/bin/bash
CUR_PWD=`pwd`
echo $CUR_PWD
BIN_PATH=${CUR_PWD}/build/bin
LIB_PATH=${CUR_PWD}/build/lib
echo $BIN_PATH
echo $LIB_PATH
export PATH=${BIN_PATH}:${PATH}
export LD_LIBRARY_PATH=${LIB_PATH}:${LD_LIBRARY_PATH}

#add third_party llvm lib and bin
LLVM_PATH=${CUR_PWD}/third-party/llvm-project/
LLVM_PATH_BIN=${LLVM_PATH}/build/bin
LLVM_PATH_LIB=${LLVM_PATH}/build/lib

export PATH=${LLVM_PATH_BIN}:$PATH
export LD_LIBRARY_PATH=${LLVM_PATH_LIB}:${LD_LIBRARY_PATH}

export PYBIND_PATH=$LIB_PATH
export PYTHONPATH=${PYBIND_PATH}:$PYTHONPATH