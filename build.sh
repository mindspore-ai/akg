#!/bin/bash
set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-v] [-p] [-j[n]]"
  echo ""
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -v Soc version. (Default: Ascend910B,Ascend310P)"
  echo "    -p The absolute path to the directory of the operator that needs to be compiled, use ',' to split. (Default: all operators)"
  echo "    -j[n] Set the threads when building (Default: half avaliable cpus)"
  echo "    -h Help"
}

# check and set options
process_options()
{
  # Process the options
  while getopts 'dv:p:j:h' opt
  do
    case "${opt}" in
      d)
        export DEBUG_MODE="on" ;;
      v)
        export SOC_VERSION="$OPTARG" ;;
      p)
        export OP_DIRS="$OPTARG" ;;
      j)
        export CMAKE_THREAD_NUM=$OPTARG ;;
      h)
        usage
        exit 0;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

process_options $@

echo "Start build."
rm -rf ./build
rm -rf ./dist
python setup.py clean --all
python setup.py bdist_wheel
echo "Finish build."
