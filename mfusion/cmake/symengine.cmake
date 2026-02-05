if(NOT COMMAND akg_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

set(SYMENGINE_URL "https://gitee.com/mirrors/SymEngine/repository/archive/v0.11.2.tar.gz")
set(SYMENGIN_SHA256 "b944dd331ba0d9ee1f1411912937e111ed7039e71264c9791b0de7e543a32ee6")

set(SYMENGINE_CMAKE_OPTIONS
    -DHAVE_SYMENGINE_NOEXCEPT:BOOL=OFF
    -DWITH_BFD:BOOL=OFF
    -DWITH_SYMENGINE_ASSERT:BOOL=OFF
    -DWITH_SYMENGINE_RCP:BOOL=ON
    -DWITH_SYMENGINE_THREAD_SAFE:BOOL=OFF
    -DWITH_SYMENGINE_THREAD_SAFE:BOOL=OFF
    -DWITH_ECM:BOOL=OFF
    -DBUILD_TESTS:BOOL=OFF
    -DBUILD_BENCHMARKS:BOOL=OFF
    -DBUILD_BENCHMARKS_GOOGLE:BOOL=OFF
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
)

akg_add_pkg(symengine
        VER 0.11.2
        URL ${SYMENGINE_URL}
        SHA256 ${SYMENGIN_SHA256}
        LIBS symengine
        CMAKE_OPTION ${SYMENGINE_CMAKE_OPTIONS})

message(STATUS "symengine_inc :${symengine_INC}")
include_directories(${symengine_INC})
add_library(mfusion::symengine ALIAS symengine::symengine)

# Since we are using static SymEngine, we must explicitly link its dependency (GMP)
# to avoid undefined references.
find_library(GMP_LIB gmp)
if(GMP_LIB)
  target_link_libraries(symengine::symengine INTERFACE ${GMP_LIB})
endif()

find_library(GMPXX_LIB gmpxx)
if(GMPXX_LIB)
  target_link_libraries(symengine::symengine INTERFACE ${GMPXX_LIB})
endif()
