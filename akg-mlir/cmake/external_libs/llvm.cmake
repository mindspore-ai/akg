#  if(USE_CUDA)
#      set(llvm_project_CXXFLAGS "-DLLVM_TARGETS_TO_BUILD='host;Native;NVPTX' -DMLIR_ENABLE_CUDA_RUNNER=ON")
#  else()
#      set(llvm_project_CXXFLAGS "-DLLVM_TARGETS_TO_BUILD='host'")
#  endif()

# if(USE_CUDA)
#     set(LLVM_TARGET_FLAGS "-DLLVM_TARGETS_TO_BUILD='host;Native;NVPTX' -DMLIR_ENABLE_CUDA_RUNNER=ON")
#  else()
#     set(LLVM_TARGET_FLAGS "-DLLVM_TARGETS_TO_BUILD='host'")
#  endif()

if (DEFINED ENV{MSLIBS_CACHE_PATH})
    set(_MS_LIB_CACHE  $ENV{MSLIBS_CACHE_PATH})
else()
    set(_MS_LIB_CACHE ${CMAKE_BINARY_DIR}/.mslib)
endif ()
message("MS LIBS CACHE PATH:  ${_MS_LIB_CACHE}")

if (NOT EXISTS ${_MS_LIB_CACHE})
    file(MAKE_DIRECTORY ${_MS_LIB_CACHE})
endif ()

if(USE_CUDA)
    set(LLVM_TARGET "gpu")
else()
    set(LLVM_TARGET "cpu")
endif()



if(ENABLE_GITEE)
    set(LLVM_URL "https://gitee.com/mirrors/llvm-project/repository/archive/llvmorg-16.0.6.tar.gz")
    set(LLVM_MD5 "2a26ad101fd9b40c6ed1bcd2e6a7e0b6")
else()
    set(LLVM_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-16.0.6.tar.gz")
    set(LLVM_MD5 "2a26ad101fd9b40c6ed1bcd2e6a7e0b6")
endif()



function(_build_llvm pkg_name)
    set(options )
    set(oneValueArgs VER URL MD5 CMAKE_PATH)
    set(multiValueArgs CMAKE_OPTION  CONFIGURE_COMMAND EXE LIBS PATCHES)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(NOT PKG_LIB_PATH)
        set(PKG_LIB_PATH lib)
    endif()

    set(__FIND_PKG_NAME ${pkg_name})
    string(TOLOWER ${pkg_name} pkg_name)
    message("pkg name:${__FIND_PKG_NAME},${pkg_name}")

    # strip directory variables to ensure third party packages are installed in consistent locations
    string(REPLACE ${CMAKE_SOURCE_DIR} "" ARGN_STRIPPED ${ARGN})
    message("REPLACE ${_MS_LIB_CACHE} "" ARGN_STRIPPED ${ARGN_STRIPPED}")
    string(REPLACE ${_MS_LIB_CACHE} "" ARGN_STRIPPED ${ARGN_STRIPPED})
    set(${pkg_name}_CONFIG_TXT
            "${pkg_name}-${PKGVER}-${LLVM_URL}-${LLVM_MD5}")
    string(MD5 ${pkg_name}_CONFIG_HASH ${${pkg_name}_CONFIG_TXT})

    message("${pkg_name} config hash: ${${pkg_name}_CONFIG_HASH}")
    # Generate hash for current pkg end

    set(${pkg_name}_BASE_DIR ${_MS_LIB_CACHE}/${pkg_name}_${${pkg_name}_CONFIG_HASH})
    set(${pkg_name}_DIRPATH ${${pkg_name}_BASE_DIR} CACHE STRING INTERNAL)
    set(LLVM_BUILD_PATH "${${pkg_name}_BASE_DIR}" PARENT_SCOPE)
    message("set LLVM_BUILD_PATH: ${LLVM_BUILD_PATH}")

    set(${__FIND_PKG_NAME}_ROOT ${${pkg_name}_BASE_DIR})
    set(${__FIND_PKG_NAME}_ROOT ${${pkg_name}_BASE_DIR} PARENT_SCOPE)

    message(" __find_pkg_then_add_target(${pkg_name} ${PKG_EXE} ${PKG_LIB_PATH} ${PKG_LIBS})")
    __find_pkg_then_add_target(${pkg_name} ${PKG_EXE} ${PKG_LIB_PATH} ${PKG_LIBS})
    if(${pkg_name}_LIBS)
        set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/include PARENT_SCOPE)
        message("Found libs: ${${pkg_name}_LIBS}")
        return()
    endif()
    
    # Download pkg
    if(NOT PKG_DIR)
	    __download_pkg(${pkg_name} ${PKG_URL} ${PKG_MD5})
    else()
        # Check if pkg is valid
        if(NOT EXISTS ${PKG_DIR})
            message(FATAL_ERROR "${PKG_DIR} not exits")
        endif()
        # If pkg is a directory, then use this directory directly
        if(IS_DIRECTORY ${PKG_DIR})
            set(${pkg_name}_SOURCE_DIR ${PKG_DIR})
        else()
            # Else, if pkg is a compressed file, decompress it first, then use the decompressed directory
            set(DECOMPRESS_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/${pkg_name})
            if(EXISTS ${DECOMPRESS_DIR})
                file(REMOVE_RECURSE ${DECOMPRESS_DIR})
            endif()
            file(MAKE_DIRECTORY ${DECOMPRESS_DIR})
            message(STATUS "Decompressing ${PKG_DIR}")
            if(${PKG_DIR} MATCHES ".tar.gz$")
                execute_process(COMMAND tar -zxf ${PKG_DIR} -C ${DECOMPRESS_DIR}
                        RESULT_VARIABLE DECOMPRESS_RESULT)
                if(NOT DECOMPRESS_RESULT EQUAL 0)
                    message(FATAL_ERROR "Decompress failed: ${PKG_DIR}")
                endif()
            else()
                message(FATAL_ERROR "pkg can only be a directory or a .tar.gz file now, but got: ${PKG_DIR}")
            endif()
            FILE(GLOB ALL_FILES ${DECOMPRESS_DIR}/*)
            list(GET ALL_FILES 0 ${pkg_name}_SOURCE_DIR)
        endif()
    endif()
    message("${pkg_name}_SOURCE_DIR : ${${pkg_name}_SOURCE_DIR}")

    # Copy pkg to the build directory and uses the copied one
    set(${pkg_name}_PATCHED_DIR ${CMAKE_BINARY_DIR}/${pkg_name})
    if(EXISTS ${${pkg_name}_PATCHED_DIR})
        file(REMOVE_RECURSE ${${pkg_name}_PATCHED_DIR})
    endif()
    file(MAKE_DIRECTORY "${${pkg_name}_PATCHED_DIR}")
    file(COPY ${${pkg_name}_SOURCE_DIR}/ DESTINATION ${${pkg_name}_PATCHED_DIR})
    set(${pkg_name}_SOURCE_DIR ${${pkg_name}_PATCHED_DIR})
    message("${pkg_name}_SOURCE_DIR : ${${pkg_name}_SOURCE_DIR}")

    # Apply patches on pkg
    foreach(_PATCH_FILE ${PKG_PATCHES})
        get_filename_component(_PATCH_FILE_NAME ${_PATCH_FILE} NAME)
        set(_LF_PATCH_FILE ${CMAKE_BINARY_DIR}/_ms_patch/${_PATCH_FILE_NAME})
        configure_file(${_PATCH_FILE} ${_LF_PATCH_FILE} NEWLINE_STYLE LF)
        message("patching ${${pkg_name}_SOURCE_DIR} -p1 < ${_LF_PATCH_FILE}")
        execute_process(COMMAND patch -p1 INPUT_FILE ${_LF_PATCH_FILE}
                        WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR}
                        RESULT_VARIABLE Result)
        if(NOT Result EQUAL "0")
            message(FATAL_ERROR "Failed patch: ${_LF_PATCH_FILE}")
        endif()
    endforeach(_PATCH_FILE)

    file(LOCK ${${pkg_name}_BASE_DIR} DIRECTORY GUARD FUNCTION RESULT_VARIABLE ${pkg_name}_LOCK_RET TIMEOUT 600)
    if(NOT ${pkg_name}_LOCK_RET EQUAL "0")
        message(FATAL_ERROR "error! when try lock ${${pkg_name}_BASE_DIR} : ${${pkg_name}_LOCK_RET}")
    endif()

    include(ProcessorCount)
    ProcessorCount(N)
    if (JOBS)
        set(THNUM ${JOBS})
    else()
        set(JOBS 8)
        if (${JOBS} GREATER ${N})
            set(THNUM ${N})
        else()
            set(THNUM ${JOBS})
        endif()
    endif ()
    message("set make thread num: ${THNUM}")
    message("execute_process(COMMAND bash ${AKG_MLIR_SOURCE_DIR}/../script/build_llvm.sh -e ${LLVM_TARGET} -j${THNUM} -s ${${pkg_name}_SOURCE_DIR}
    -t ${${pkg_name}_BASE_DIR})")
    set(THNUM 32)
    execute_process(COMMAND bash ${AKG_MLIR_SOURCE_DIR}/../script/build_llvm.sh -e "${LLVM_TARGET}" 
        -j${THNUM} -s ${${pkg_name}_SOURCE_DIR} -t ${${pkg_name}_BASE_DIR})
endfunction()


_build_llvm(llvm_project
         VER 16.0.6
         LIBS mlir_async_runtime clang
         EXE mlir-opt
         URL ${LLVM_URL}
         MD5 ${LLVM_MD5}
         PATCHES ${AKG_SOURCE_DIR}/third-party/llvm_patch_7cbf1a2591520c2491aa35339f227775f4d3adf6.patch
        )


include_directories(${Symengine_INC})
add_library(akg::mlir_async_runtime ALIAS llvm_project::mlir_async_runtime)
add_library(akg::mlir_async_runtime ALIAS llvm_project::clang)






