include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

if (CMAKE_SYSTEM_NAME MATCHES "Windows" AND ${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.17.0)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .dll ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif ()

function(akg_add_submodule_obj des_submodule_objs sub_dir submodule_name_obj)

    add_subdirectory(${sub_dir})

    if(NOT TARGET ${submodule_name_obj})
        message(FATAL_ERROR "Can not find submodule '${submodule_name_obj}'. in ${CMAKE_CURRENT_LIST_FILE}")
    endif()
    if("$<TARGET_OBJECTS:${submodule_name_obj}>" IN_LIST ${des_submodule_objs})
        message(FATAL_ERROR "submodule '${submodule_name_obj}' added more than once. in ${CMAKE_CURRENT_LIST_FILE}")
    endif()

    set(${des_submodule_objs} ${${des_submodule_objs}} $<TARGET_OBJECTS:${submodule_name_obj}> PARENT_SCOPE)

endfunction()

if (DEFINED ENV{MSLIBS_CACHE_PATH})
    set(_MS_LIB_CACHE  $ENV{MSLIBS_CACHE_PATH})
else()
    set(_MS_LIB_CACHE ${CMAKE_BINARY_DIR}/.mslib)
endif ()
message("MS LIBS CACHE PATH:  ${_MS_LIB_CACHE}")

if (NOT EXISTS ${_MS_LIB_CACHE})
    file(MAKE_DIRECTORY ${_MS_LIB_CACHE})
endif ()

if (DEFINED ENV{MSLIBS_SERVER})
    set(LOCAL_LIBS_SERVER  $ENV{MSLIBS_SERVER})
    message("LOCAL_LIBS_SERVER:  ${LOCAL_LIBS_SERVER}")
endif ()

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

if(LOCAL_LIBS_SERVER)
    if (NOT ENV{no_proxy})
        set(ENV{no_proxy} "${LOCAL_LIBS_SERVER}")
    else()
        string(FIND $ENV{no_proxy} ${LOCAL_LIBS_SERVER} IP_POS)
        if (${IP_POS} EQUAL -1)
            set(ENV{no_proxy} "$ENV{no_proxy},${LOCAL_LIBS_SERVER}")
        endif ()
    endif ()
endif()

function(__download_pkg pkg_name pkg_url pkg_md5)

    if(LOCAL_LIBS_SERVER)
        get_filename_component(_URL_FILE_NAME ${pkg_url} NAME)
        set(pkg_url "http://${LOCAL_LIBS_SERVER}:8081/libs/${pkg_name}/${_URL_FILE_NAME}" ${pkg_url})
    endif()

    FetchContent_Declare(
            ${pkg_name}
            URL      ${pkg_url}
            URL_HASH MD5=${pkg_md5}
    )
    FetchContent_GetProperties(${pkg_name})
    message("download: ${${pkg_name}_SOURCE_DIR} , ${pkg_name} , ${pkg_url}")
    if(NOT ${pkg_name}_POPULATED)
        FetchContent_Populate(${pkg_name})
        set(${pkg_name}_SOURCE_DIR ${${pkg_name}_SOURCE_DIR} PARENT_SCOPE)
    endif()

endfunction()

function(__download_pkg_with_git pkg_name pkg_url pkg_git_commit pkg_md5)

    if(LOCAL_LIBS_SERVER)
        set(pkg_url "http://${LOCAL_LIBS_SERVER}:8081/libs/${pkg_name}/${pkg_git_commit}")
        FetchContent_Declare(
                ${pkg_name}
                URL      ${pkg_url}
                URL_HASH MD5=${pkg_md5}
    )
    else()
	FetchContent_Declare(
            ${pkg_name}
	    GIT_REPOSITORY      ${pkg_url}
	    GIT_TAG             ${pkg_git_commit})
    endif()
    FetchContent_GetProperties(${pkg_name})
    message("download: ${${pkg_name}_SOURCE_DIR} , ${pkg_name} , ${pkg_url}")
    if(NOT ${pkg_name}_POPULATED)
        FetchContent_Populate(${pkg_name})
        set(${pkg_name}_SOURCE_DIR ${${pkg_name}_SOURCE_DIR} PARENT_SCOPE)
    endif()

endfunction()


function(__find_pkg_then_add_target pkg_name pkg_exe lib_path)

    unset(${pkg_name}_LIBS)

    message("_FIND:${${pkg_name}_BASE_DIR}")

    if(pkg_exe)
        find_program(${pkg_exe}_EXE ${pkg_exe} PATHS ${${pkg_name}_BASE_DIR}/bin NO_DEFAULT_PATH)
        if(NOT ${pkg_exe}_EXE)
            return()
        endif()
        add_executable(${pkg_name}::${pkg_exe} IMPORTED GLOBAL)
        set_target_properties(${pkg_name}::${pkg_exe} PROPERTIES
                IMPORTED_LOCATION ${${pkg_exe}_EXE}
                )
        message("found ${${pkg_exe}_EXE}")
    endif()

    foreach(_LIB_NAME ${ARGN})
        set(_LIB_SEARCH_NAME ${_LIB_NAME})
        set(_LIB_TYPE SHARED)
        if (${pkg_name}_USE_STATIC_LIBS)
            set(_LIB_SEARCH_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
            set(_LIB_TYPE STATIC)
        endif ()
        set(${_LIB_NAME}_LIB ${_LIB_NAME}_LIB-NOTFOUND)
        find_library(${_LIB_NAME}_LIB ${_LIB_SEARCH_NAME} PATHS ${${pkg_name}_BASE_DIR}/${lib_path} NO_DEFAULT_PATH)

        if(NOT ${_LIB_NAME}_LIB)
            return()
        endif()

        add_library(${pkg_name}::${_LIB_NAME} ${_LIB_TYPE} IMPORTED GLOBAL)
        if (WIN32 AND ${_LIB_TYPE} STREQUAL "SHARED")
            set_target_properties(${pkg_name}::${_LIB_NAME} PROPERTIES IMPORTED_IMPLIB_RELEASE ${${_LIB_NAME}_LIB})
        else()
            set_target_properties(${pkg_name}::${_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${${_LIB_NAME}_LIB})
        endif()

        if (EXISTS ${${pkg_name}_BASE_DIR}/include)
            set_target_properties(${pkg_name}::${_LIB_NAME} PROPERTIES 
                INTERFACE_INCLUDE_DIRECTORIES "${${pkg_name}_BASE_DIR}/include")
        endif ()

        list(APPEND ${pkg_name}_LIBS ${pkg_name}::${_LIB_NAME})
        message("found ${${_LIB_NAME}_LIB}")
        STRING( REGEX REPLACE "(.+)/(.+)" "\\1" LIBPATH ${${_LIB_NAME}_LIB})
        set(${pkg_name}_LIBPATH ${LIBPATH} CACHE STRING INTERNAL)
    endforeach(_LIB_NAME)

    set(${pkg_name}_LIBS ${${pkg_name}_LIBS} PARENT_SCOPE)
endfunction()

function(__exec_cmd)
    set(options )
    set(oneValueArgs WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)

    cmake_parse_arguments(EXEC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    execute_process(COMMAND ${EXEC_COMMAND}
            WORKING_DIRECTORY ${EXEC_WORKING_DIRECTORY}
            RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL "0")
        message(FATAL_ERROR "error! when ${EXEC_COMMAND} in ${EXEC_WORKING_DIRECTORY}")
    endif()
endfunction()

function(__check_patches pkg_patches)
    # check patches
    if (PKG_PATCHES)
        file(TOUCH ${_MS_LIB_CACHE}/${pkg_name}_patch.md5)
        file(READ ${_MS_LIB_CACHE}/${pkg_name}_patch.md5 ${pkg_name}_PATCHES_MD5)

        message("patches md5:${${pkg_name}_PATCHES_MD5}")

        set(${pkg_name}_PATCHES_NEW_MD5 )
        foreach(_PATCH ${PKG_PATCHES})
            file(MD5 ${_PATCH} _PF_MD5)
            set(${pkg_name}_PATCHES_NEW_MD5 "${${pkg_name}_PATCHES_NEW_MD5},${_PF_MD5}")
        endforeach(_PATCH)

        if (NOT ${pkg_name}_PATCHES_MD5 STREQUAL ${pkg_name}_PATCHES_NEW_MD5)
            set(${pkg_name}_PATCHES ${PKG_PATCHES})
            file(REMOVE_RECURSE "${_MS_LIB_CACHE}/${pkg_name}-subbuild")
            file(WRITE ${_MS_LIB_CACHE}/${pkg_name}_patch.md5 ${${pkg_name}_PATCHES_NEW_MD5})
            message("patches changed : ${${pkg_name}_PATCHES_NEW_MD5}")
        endif ()
    endif ()
endfunction()

set(AKG_FIND_NO_DEFAULT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH
                            NO_CMAKE_BUILDS_PATH NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PATH
                            NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)

function(akg_add_pkg pkg_name )
    set(options )
    set(oneValueArgs URL MD5 GIT_REPOSITORY GIT_TAG VER EXE DIR CMAKE_PATH CUSTOM_CMAKE)
    set(multiValueArgs CMAKE_OPTION LIBS PATCHES)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(NOT PKG_LIB_PATH)
        set(PKG_LIB_PATH lib)
    endif()

    set(__FIND_PKG_NAME ${pkg_name})
    string(TOLOWER ${pkg_name} pkg_name)
    message("pkg name:${__FIND_PKG_NAME},${pkg_name}")

    # Generate hash for current pkg
    set(${pkg_name}_CUSTOM_CMAKE_HASH )
    if(PKG_CUSTOM_CMAKE)
        file(MD5 ${PKG_CUSTOM_CMAKE}/CMakeLists.txt _CUSTOM_CMAKE_MD5)
        set(${pkg_name}_CUSTOM_CMAKE_HASH "${_CUSTOM_CMAKE_MD5}")
    endif()

    set(${pkg_name}_PATCHES_HASH )
    foreach(_PATCH ${PKG_PATCHES})
        file(MD5 ${_PATCH} _PF_MD5)
        set(${pkg_name}_PATCHES_HASH "${${pkg_name}_PATCHES_HASH},${_PF_MD5}")
    endforeach(_PATCH)

    # strip directory variables to ensure third party packages are installed in consistent locations
    string(REPLACE ${CMAKE_SOURCE_DIR} "" ARGN_STRIPPED ${ARGN})
    string(REPLACE ${_MS_LIB_CACHE} "" ARGN_STRIPPED ${ARGN_STRIPPED})
    set(${pkg_name}_CONFIG_TXT
            "${CMAKE_CXX_COMPILER_VERSION}-${CMAKE_C_COMPILER_VERSION}
            ${ARGN_STRIPPED}-${${pkg_name}_USE_STATIC_LIBS}-${${pkg_name}_PATCHES_HASH}
            ${${pkg_name}_CXXFLAGS}-${${pkg_name}_CFLAGS}-${${pkg_name}_LDFLAGS}")
    string(REPLACE ";" "-" ${pkg_name}_CONFIG_TXT ${${pkg_name}_CONFIG_TXT})
    string(MD5 ${pkg_name}_CONFIG_HASH ${${pkg_name}_CONFIG_TXT})

    message("${pkg_name} config hash: ${${pkg_name}_CONFIG_HASH}")
    # Generate hash for current pkg end

    set(${pkg_name}_BASE_DIR ${_MS_LIB_CACHE}/${pkg_name}_${${pkg_name}_CONFIG_HASH})
    set(${pkg_name}_DIRPATH ${${pkg_name}_BASE_DIR} CACHE STRING INTERNAL)

    if(NOT PKG_EXE)
        set(PKG_EXE 0)
    endif()

    set(${__FIND_PKG_NAME}_ROOT ${${pkg_name}_BASE_DIR})
    set(${__FIND_PKG_NAME}_ROOT ${${pkg_name}_BASE_DIR} PARENT_SCOPE)

    # Check if pkg exists in cache, if exists, use the cached one
    if (PKG_LIBS)
        __find_pkg_then_add_target(${pkg_name} ${PKG_EXE} ${PKG_LIB_PATH} ${PKG_LIBS})
        if(${pkg_name}_LIBS)
            set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/include PARENT_SCOPE)
            message("Found libs: ${${pkg_name}_LIBS}")
            return()
        endif()
    endif()

    # Download pkg
    if(NOT PKG_DIR)
        if(PKG_GIT_REPOSITORY)
            __download_pkg_with_git(${pkg_name} ${PKG_GIT_REPOSITORY} ${PKG_GIT_TAG} ${PKG_MD5})
        else()
            message("--__download_pkg")
	    __download_pkg(${pkg_name} ${PKG_URL} ${PKG_MD5})
        endif()
        foreach(_SUBMODULE_FILE ${PKG_SUBMODULES})
            STRING( REGEX REPLACE "(.+)_(.+)" "\\1" _SUBMODEPATH ${_SUBMODULE_FILE})
            STRING( REGEX REPLACE "(.+)/(.+)" "\\2" _SUBMODENAME ${_SUBMODEPATH})
            file(GLOB ${pkg_name}_INSTALL_SUBMODULE ${_SUBMODULE_FILE}/*)
            file(COPY ${${pkg_name}_INSTALL_SUBMODULE} DESTINATION ${${pkg_name}_SOURCE_DIR}/3rdparty/${_SUBMODENAME})
        endforeach (_SUBMODULE_FILE)
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

    if(PKG_CUSTOM_CMAKE)
        file(GLOB ${pkg_name}_cmake ${PKG_CUSTOM_CMAKE}/CMakeLists.txt)
        file(COPY ${${pkg_name}_cmake} DESTINATION ${${pkg_name}_SOURCE_DIR})
    endif()

    if(${pkg_name}_SOURCE_DIR)
        if(PKG_CMAKE_OPTION)
            # in cmake
            file(MAKE_DIRECTORY ${${pkg_name}_SOURCE_DIR}/_build)
            if(${pkg_name}_CFLAGS)
                set(${pkg_name}_CMAKE_CFLAGS "-DCMAKE_C_FLAGS=${${pkg_name}_CFLAGS}")
            endif()
            if (${pkg_name}_CXXFLAGS)
                set(${pkg_name}_CMAKE_CXXFLAGS "-DCMAKE_CXX_FLAGS=${${pkg_name}_CXXFLAGS}")
            endif()

            if (${pkg_name}_LDFLAGS)
                if (${pkg_name}_USE_STATIC_LIBS)
                    set(${pkg_name}_CMAKE_LDFLAGS "-DCMAKE_STATIC_LINKER_FLAGS=${${pkg_name}_LDFLAGS}")
                else()
                    set(${pkg_name}_CMAKE_LDFLAGS "-DCMAKE_SHARED_LINKER_FLAGS=${${pkg_name}_LDFLAGS}")
                endif()
            endif()

            __exec_cmd(COMMAND ${CMAKE_COMMAND} ${PKG_CMAKE_OPTION} -G ${CMAKE_GENERATOR}
                    ${${pkg_name}_CMAKE_CFLAGS} ${${pkg_name}_CMAKE_CXXFLAGS} ${${pkg_name}_CMAKE_LDFLAGS}
                    ${${pkg_name}_CONFIG_FILE_DIR} ${${pkg_name}_DEPEND_INCLUDE_DIR} ${${pkg_name}_DEPEND_LIB_DIR}
                    -DCMAKE_INSTALL_PREFIX=${${pkg_name}_BASE_DIR} ${${pkg_name}_SOURCE_DIR}/${PKG_CMAKE_PATH}
                    WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR}/_build)

            __exec_cmd(COMMAND ${CMAKE_COMMAND} --build . --target install -- -j${THNUM}
                    WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR}/_build)
        endif()
    endif()

    if (PKG_LIBS)
        __find_pkg_then_add_target(${pkg_name} ${PKG_EXE} ${PKG_LIB_PATH} ${PKG_LIBS})
        set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/include PARENT_SCOPE)
        if(NOT ${pkg_name}_LIBS)
            message(FATAL_ERROR "Can not find pkg: ${pkg_name}")
        endif()
    else()
        find_package(${__FIND_PKG_NAME} ${PKG_VER} QUIET)
        if (${__FIND_PKG_NAME}_FOUND)
            set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/include PARENT_SCOPE)
            message("Found pkg: ${${__FIND_PKG_NAME}_LIBRARIES}")
            return()
        endif()
    endif()
endfunction()

function(find_python_package out_lib)
  # Use PYTHON_EXECUTABLE if it is defined, otherwise default to python
  if("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(PYTHON_EXECUTABLE "python3")
  else()
    set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
  endif()

  execute_process(
          COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE lib)
  string(STRIP "${lib}" lib)
  set(${out_lib} ${lib} PARENT_SCOPE)
endfunction()
