set(isl_USE_STATIC_LIBS ON)
set(isl_CONFIG_FILE_DIR "-DISL_WRAP_DIR=${AKG_SOURCE_DIR}/third_party/isl_wrap")
set(isl_DEPEND_INCLUDE_DIR "-DGMP_INCLUDE_DIR=${GMP_INCLUDE_DIR}")
set(isl_DEPEND_LIB_DIR "-DGMP_LIBRARY=${GMP_LIBRARY}")

akg_add_pkg(isl
        VER 0.22
        DIR ${AKG_SOURCE_DIR}/third_party/isl-0.22.tar.gz
        LIBS isl_fixed
        CUSTOM_CMAKE ${AKG_SOURCE_DIR}/third_party/isl_wrap
        PATCHES ${AKG_SOURCE_DIR}/third_party/patch/isl/isl.patch ${AKG_SOURCE_DIR}/third_party/patch/isl/isl-influence.patch
        CMAKE_OPTION " ")
include_directories("${AKG_SOURCE_DIR}/third_party/isl_wrap/include")
include_directories("${isl_INC}/include")
include_directories("${isl_INC}")
link_directories("${isl_LIBPATH}")
add_library(akg::isl_fixed ALIAS isl::isl_fixed)
