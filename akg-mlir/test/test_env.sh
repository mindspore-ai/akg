CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}" )" && pwd )"
if [ -z "${LLVM_HOME}" ]; then
    LLVM_HOME=${CUR_DIR}/../../third-party/llvm-project/build
fi
export PYTHONPATH=${CUR_DIR}/../python:${PYTHONPATH}
export PYTHONPATH=$LLVM_HOME/tools/mlir/python_packages/mlir_core:$PYTHONPATH
export PATH=${CUR_DIR}/../../build/akg-mlir/bin:${PATH}
export LD_LIBRARY_PATH=${CUR_DIR}/../../build/akg-mlir/lib:${LD_LIBRARY_PATH}
