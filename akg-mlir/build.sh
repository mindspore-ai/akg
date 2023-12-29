#!/bin/bash

set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
AKG_MLIR_OUTPUT_PATH=${BASE_PATH}/output
THIRD_PARTY_PATH=${BASE_PATH}/third-party
PATH_TO_SOURCE_LLVM=${THIRD_PARTY_PATH}/llvm-project/
PATH_TO_SOURCE_SYMENGINE=${THIRD_PARTY_PATH}/symengine/
PATH_TO_SOURCE_POLYTOPS=${THIRD_PARTY_PATH}/polytops/

export LD_LIBRARY_PATH=${BASE_PATH}/build/lib:${LD_LIBRARY_PATH}

_COUNT=0
define_flag() {
    local var=$1
    local default=$2
    local opt=$3
    local flag=$4
    local intro=$5

    # count flags number
    _COUNT=$((${_COUNT} + 1))

    local max_flag=8
    local mod_max=$((${_COUNT} % ${max_flag}))
    local intro_space="    "
    local flag_space="              "
    local flag2=`echo ${flag} | awk '{print $1}'`

    # set global varibles
    if [[ "X${var}" != "X" ]] && [[ "X${default}" != "X" ]]; then
        eval "${var}=${default}"
    fi
    _OPTS="${_OPTS}${opt}"
    if [[ "X${mod_max}" = "X1" ]] && [[ "X${mod_max}" != "X${_COUNT}" ]]; then
        _FLAGS="${_FLAGS}\n${flag_space}[${flag}]"
    else
        _FLAGS="${_FLAGS} [${flag}]"
    fi
    _INTROS="${_INTROS}${intro_space}${flag2} ${intro}\n"
}

# define options
define_flag ""                ""                    "h"  "-h"                     "Print usage"
define_flag CLEAN_BUILT       "off"                 "c"  "-c"                     "Clean built files, default: off"
define_flag DEBUG_MODE        "off"                 "d"  "-d"                     "Enable debug mode, default: off"
define_flag ENABLE_UNIT_TEST  "off"                 "t"  "-t"                     "Unit test: on or off, default: off"
define_flag COMPILE_MODE      "all"                 "m:" "-m"                     "Compile mode: akg-mlir-only or all, default: all"
define_flag BACKEND_ENV       "auto"                "e:" "-e"                     "Backend Environment: cpu, gpu, or auto, default: auto"
define_flag DEPENDENT_BUILD   "none"                "S:"  "-S"                    "Specifies the build path of third-partys, default: none \n\t[0]llvm-project\n\t[1]symengine\n\t[2]polytops"
define_flag DEPENDENT_SOURCE  "none"                "s:"  "-s"                    "Specifies the source path of third-partys, default: none \n\t[0]llvm-project\n\t[1]symengine\n\t[2]polytops"
define_flag UPDATE_SUBMODULE  "off"                 "u"  "-u"                     "Update submodule, default: off"
define_flag THREAD_NUM        8                     "j:" "-j[n]"                  "Set the threads number when building, default: -j8"

# print usage message
usage() {
    printf "Usage:\nbash build.sh${_FLAGS}\n\n"
    printf "Options:\n${_INTROS}"
}

# check and set options
checkopts() {
    # Process the options
    while getopts "${_OPTS}" opt
    do
        OPTARGRAW=${OPTARG}
        OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
        case "${opt}" in
            h)
                usage
                exit 0
                ;;
            m)
                COMPILE_MODE=${OPTARG}
                ;;
            c)
                CLEAN_BUILT="on"
                ;;
            d)
                DEBUG_MODE="on"
                ;;
            u)
                UPDATE_SUBMODULE="on"
                ;;
            j)
                THREAD_NUM=${OPTARG}
                ;;
            e)
                BACKEND_ENV=${OPTARG}
                ;;
            S)
                DEPENDENT_BUILD=${OPTARG}
                ;;
            s)
                DEPENDENT_SOURCE=${OPTARG}
                ;;
            t)
                ENABLE_UNIT_TEST="on"
                ;;
            *)
                echo "Unknown option ${opt}!"
                usage
                exit 1;
        esac
    done
}
checkopts "$@"

if [[ "X${DEBUG_MODE}" = "Xon" ]]; then
  _BUILD_TYPE="Debug"
else
  _BUILD_TYPE="Release"
fi

if [[ "X${BACKEND_ENV}" = "XGPU" ]] || [[ "X${BACKEND_ENV}" = "Xgpu" ]]; then
    CUDA_BACKEND="ON"
elif [[ "X${BACKEND_ENV}" = "XAUTO" ]] || [[ "X${BACKEND_ENV}" = "Xauto" ]]; then
    CUDA_BACKEND="AUTO"
else
    CUDA_BACKEND="OFF"
fi

THIRD_PARTY_COUNT=0
if [[ "X${DEPENDENT_SOURCE}" != "Xnone" ]]; then
    for local_var in ${DEPENDENT_SOURCE[@]}
    do
        if [[ $THIRD_PARTY_COUNT -eq 0 ]] && [[ "X${local_var}" != "Xnone" ]]; then
            export PATH_TO_SOURCE_LLVM=${local_var}
        fi
        if [[ $THIRD_PARTY_COUNT -eq 1 ]] && [[ "X${local_var}" != "Xnone" ]]; then
            export PATH_TO_SOURCE_SYMENGINE=${local_var}
        fi
        if [[ $THIRD_PARTY_COUNT -eq 2 ]] && [[ "X${local_var}" != "Xnone" ]]; then
            export PATH_TO_SOURCE_POLYTOPS=${local_var}
        fi
        let THIRD_PARTY_COUNT+=1
    done
fi

update_submodule(){
  git -C "${BASE_PATH}" submodule update --init --depth 1
}

third_party_patch() {
  echo "Start patching to llvm."
  local FILE=${THIRD_PARTY_PATH}/llvm_patch_7cbf1a2591520c2491aa35339f227775f4d3adf6.patch
  if [ -f "$FILE" ]; then
    cd ${PATH_TO_SOURCE_LLVM}
    local LLVM_CUR_COMMIT_ID=$(echo `git rev-parse HEAD`)
    if [[ "X${LLVM_CUR_COMMIT_ID}" != "X7cbf1a2591520c2491aa35339f227775f4d3adf6" ]]; then
        git checkout main
        git checkout .
        git clean -df
        git pull
        git reset --hard 7cbf1a2591520c2491aa35339f227775f4d3adf6
        echo "set llvm to commit: 7cbf1a2591520c2491aa35339f227775f4d3adf6"
    fi
    git checkout .
    git clean -df
    patch -p1 -i ${FILE}
    echo "Success patch to llvm!"
  fi
}

build_llvm() {
    echo "Start building llvm project."
    LLVM_BASE_PATH=${PATH_TO_SOURCE_LLVM}
    echo "LLVM_BASE_PATH = ${PATH_TO_SOURCE_LLVM}"
    cd ${LLVM_BASE_PATH}
    if [ ! -d "./build" ]; then
        mkdir -pv build
    fi
    LLVM_BUILD_PATH=${LLVM_BASE_PATH}/build
    echo "LLVM_BUILD_PATH = ${LLVM_BUILD_PATH}"
    cd ${LLVM_BUILD_PATH}
    local LLVM_CMAKE_ARGS="-G Ninja "
    if [[ "X${BACKEND_ENV}" = "XGPU" ]] || [[ "X${BACKEND_ENV}" = "Xgpu" ]]; then
        LLVM_CMAKE_ARGS="${LLVM_CMAKE_ARGS} -DLLVM_TARGETS_TO_BUILD='host;Native;NVPTX'"
        LLVM_CMAKE_ARGS="${LLVM_CMAKE_ARGS} -DMLIR_ENABLE_CUDA_RUNNER=ON"
    elif [[ "X${BACKEND_ENV}" = "XAUTO" ]] || [[ "X${BACKEND_ENV}" = "Xauto" ]]; then
        local flag=$(whereis cuda)
        if [[ "${flag}" != "cuda:" ]]; then
            echo "CUDA environment found"
            LLVM_CMAKE_ARGS="${LLVM_CMAKE_ARGS} -DLLVM_TARGETS_TO_BUILD='host;Native;NVPTX'"
            LLVM_CMAKE_ARGS="${LLVM_CMAKE_ARGS} -DMLIR_ENABLE_CUDA_RUNNER=ON"
        fi
    else
        LLVM_CMAKE_ARGS="${LLVM_CMAKE_ARGS} -DLLVM_TARGETS_TO_BUILD='host'"
    fi

    cmake ../llvm \
    ${LLVM_CMAKE_ARGS} \
    -DPython3_FIND_STRATEGY=LOCATION \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang;openmp" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_BUILD_TYPE=${_BUILD_TYPE} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

    export PATH_TO_BUILT_LLVM=${PWD}
    cmake --build . --config ${_BUILD_TYPE} -j${THREAD_NUM}
    cmake --install ${LLVM_BUILD_PATH} --component clang --prefix ${AKG_MLIR_OUTPUT_PATH}
    cmake --install ${LLVM_BUILD_PATH} --component llc --prefix ${AKG_MLIR_OUTPUT_PATH}
    echo "Success to build llvm project!"
}

build_symengine() {
    echo "Start building symengine project."
    SYMENGINE_BASE_PATH=${PATH_TO_SOURCE_SYMENGINE}
    echo "SYMENGINE_BASE_PATH = ${SYMENGINE_BASE_PATH}"
    cd ${SYMENGINE_BASE_PATH}
    if [ ! -d "./build" ]; then
        mkdir -pv build
    fi
    SYMENGINE_BUILD_PATH=${SYMENGINE_BASE_PATH}/build
    echo "SYMENGINE_BUILD_PATH = ${SYMENGINE_BUILD_PATH}"
    cd ${SYMENGINE_BUILD_PATH}
    cmake .. \
    -DHAVE_SYMENGINE_NOEXCEPT=OFF \
    -DCMAKE_BUILD_TYPE:STRING=${_BUILD_TYPE} \
    -DWITH_BFD:BOOL=OFF \
    -DWITH_SYMENGINE_ASSERT:BOOL=OFF \
    -DWITH_SYMENGINE_RCP:BOOL=ON \
    -DWITH_SYMENGINE_THREAD_SAFE:BOOL=OFF \
    -DWITH_ECM:BOOL=OFF \
    -DBUILD_TESTS:BOOL=OFF \
    -DBUILD_BENCHMARKS:BOOL=OFF \
    -DBUILD_BENCHMARKS_GOOGLE:BOOL=OFF \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
    make -j${THREAD_NUM}
    export PATH_TO_BUILT_SYMENGINE=${PWD}
    echo "Success to build symengine project!"
}

build_akg_mlir() {
    echo "Start building akg_mlir project."
    cd ${BASE_PATH}
    if [ ! -d "./build" ]; then
        mkdir -pv build
    fi
    AKG_MLIR_BUILD_PATH=${BASE_PATH}/build
    cd ${AKG_MLIR_BUILD_PATH}
    if [[ "X${CLEAN_BUILT}" = "Xon" ]]; then
        cmake --build . --target clean
        echo "Success to clean akg-mlir built files."
    fi

    local -a AKG_MLIR_ARGS=( )
    if [[ "x${AKG_MLIR_POLYTOPS_GIT_REPOSITORY}" != "x" ]]; then
      AKG_MLIR_ARGS+=( "-DAKG_MLIR_POLYTOPS_SOURCE=git" )
      AKG_MLIR_ARGS+=( "-DAKG_MLIR_POLYTOPS_GIT_REPOSITORY=${AKG_MLIR_POLYTOPS_GIT_REPOSITORY}" )
    fi

    cmake ../compiler/cmake/ \
    "${AKG_MLIR_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=${_BUILD_TYPE} \
    -DLLVM_BUILD_PATH=${PATH_TO_BUILT_LLVM} \
    -DSYMENGINE_BUILD_PATH=${PATH_TO_BUILT_SYMENGINE} \
    -DLLVM_EXTERNAL_LIT=${PATH_TO_BUILT_LLVM}/bin/llvm-lit \
    -DUSE_CUDA=${CUDA_BACKEND} \
    -Wno-dev
    export PATH=${AKG_MLIR_BUILD_PATH}/bin:$PATH
	export LD_LIBRARY_PATH=${AKG_MLIR_BUILD_PATH}/lib:${LD_LIBRARY_PATH}
    if [[ "X${ENABLE_UNIT_TEST}" = "Xon" ]]; then
        cmake --build . --config ${_BUILD_TYPE} -j${THREAD_NUM} --target check-akg-mlir
    else
        cmake --build . --config ${_BUILD_TYPE} -j${THREAD_NUM}
    fi
    echo "Success to build akg_mlir project!"
}

echo "---------------- akg-mlir: build start ----------------"
if [[ "X${COMPILE_MODE}" = "Xakg-mlir-only" ]] || [[ "X${DEPENDENT_BUILD}" != "Xnone" ]]; then
    export PATH_TO_BUILT_LLVM=${BASE_PATH}/third-party/llvm-project/build
    export PATH_TO_BUILT_SYMENGINE=${BASE_PATH}/third-party/symengine/build
    export PATH_TO_BUILT_POLYTOPS=${BASE_PATH}/third-party/polytops/build
    THIRD_PARTY_COUNT=0
    if [[ "X${DEPENDENT_BUILD}" != "Xnone" ]]; then
        for local_var in ${DEPENDENT_BUILD[@]}
        do
            if [[ $THIRD_PARTY_COUNT -eq 0 ]] && [[ "X${local_var}" != "Xnone" ]]; then
                export PATH_TO_BUILT_LLVM=${local_var}
            fi
            if [[ $THIRD_PARTY_COUNT -eq 1 ]] && [[ "X${local_var}" != "Xnone" ]]; then
                export PATH_TO_BUILT_SYMENGINE=${local_var}
            fi
            if [[ $THIRD_PARTY_COUNT -eq 2 ]] && [[ "X${local_var}" != "Xnone" ]]; then
                export PATH_TO_BUILT_POLYTOPS=${local_var}
            fi
            let THIRD_PARTY_COUNT+=1
        done
    fi
    build_akg_mlir
else
    if [[ "X${UPDATE_SUBMODULE}" = "Xon" ]]; then
        update_submodule
    fi
    third_party_patch
    build_llvm
    build_symengine
    build_akg_mlir
fi

echo "---------------- akg-mlir: build end ----------------"
