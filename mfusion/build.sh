#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

MFUSION_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
BUILD_DIR="${MFUSION_DIR}/build"
# Parse arguments
THREAD_NUM=$(nproc)
BUILD_TYPE="Release"
BUILD_TESTS="OFF"

usage()
{
    echo "Usage:"
    echo "bash build.sh [-d] [-h] [-i] [-j[n]] [-s path] [-t]"
    echo ""
    echo "Options:"
    echo "    -d Debug mode"
    echo "    -h Print usage"
    echo "    -i Incremental build"
    echo "    -j[n] Set the threads when building (Default: the number of cpu)"
    echo "    -s Specifies the CMAKE_PREFIX_PATH for dependencies"
    echo "    -t Enable unit test (Default: disable)"
}

while getopts 'dhij:s:t' opt
do
    case "${opt}" in
        d)
            BUILD_TYPE=Debug
            ;;
        h)
            usage
            exit 0
            ;;
        i)
            INC_BUILD=1
            ;;
        j)
            THREAD_NUM=${OPTARG}
            ;;
        s)
            PREFIX_PATH=${OPTARG}
            ;;
        t)
            BUILD_TESTS="ON"
            ;;
        *)
            echo "Unknown option ${opt}!"
            usage
            exit 1
    esac
done

##################################################
# Install build dependencies
##################################################
python -c "import build" 2>/dev/null || {
    echo "Installing Python build package..."
    pip install build
}

# packaging>=24.2
python -c "
import packaging
from packaging.version import parse
assert parse(packaging.__version__) >= parse('24.2'), f'packaging {packaging.__version__} < 24.2'
" 2>/dev/null || pip install "packaging>=24.2"


# Clean build directory if not incremental build
if [[ "X$INC_BUILD" != "X1" ]]; then
    if [[ -d "${BUILD_DIR}" ]]; then
        echo "Removing build directory for clean build"
        rm -rf "${BUILD_DIR}"
    fi
fi

# Create directories
mkdir -pv "${BUILD_DIR}"

echo "---------------- MFusion: build start ----------------"

# Set environment variables
export BUILD_JOBS=${THREAD_NUM}
export BUILD_TYPE=${BUILD_TYPE}
export BUILD_TESTS=${BUILD_TESTS}
if [[ -n "${PREFIX_PATH}" ]]; then
    export CMAKE_PREFIX_PATH="${PREFIX_PATH}"
fi
if [[ "X$INC_BUILD" = "X1" ]]; then
    export INC_BUILD=1
else
    unset INC_BUILD
fi

# Build with python -m build
set -x
python -m build --wheel --no-isolation
set +x

# Move the built wheel to the output directory
mkdir -p output
cp dist/*.whl output/
rm -rf dist

echo "---------------- MFusion: build end ----------------"
