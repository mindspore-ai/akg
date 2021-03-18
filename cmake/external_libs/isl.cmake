set(isl_USE_STATIC_LIBS ON)
set(isl_CONFIG_FILE_DIR "-DISL_WRAP_DIR=${AKG_SOURCE_DIR}/third_party/isl_wrap")
set(isl_DEPEND_INCLUDE_DIR "-DGMP_INCLUDE_DIR=${GMP_INCLUDE_DIR}")
set(isl_DEPEND_LIB_DIR "-DGMP_LIBRARY=${GMP_LIBRARY}")

if(ENABLE_GITEE)
    set(ISL_URL "https://gitee.com/mirrors/isl/repository/archive/isl-0.22?format=tar.gz")
    set(ISL_MD5 "2098cec031b46c5837404a0de0e0b8ba")
else()
    set(ISL_URL "http://isl.gforge.inria.fr/isl-0.22.tar.gz")
    set(ISL_MD5 "671d0a5e10467a5c6db0893255278845")
endif()

akg_add_pkg(isl
        VER 0.22
        LIBS isl_fixed
        URL ${ISL_URL}
        MD5 ${ISL_MD5}
        CUSTOM_CMAKE ${AKG_SOURCE_DIR}/third_party/isl_wrap
        PATCHES ${AKG_SOURCE_DIR}/third_party/patch/isl/isl.patch
        CMAKE_OPTION " ")
include_directories("${AKG_SOURCE_DIR}/third_party/isl_wrap/include")
include_directories("${isl_INC}/include")
include_directories("${isl_INC}")
link_directories("${isl_LIBPATH}")
add_library(akg::isl_fixed ALIAS isl::isl_fixed)
