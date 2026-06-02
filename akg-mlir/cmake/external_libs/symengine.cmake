
if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
  set(REQ_URL "https://gitee.com/mirrors/SymEngine/repository/archive/v0.14.0.tar.gz")
  set(SHA256 "2885e21a8498fa4d13584b1a10367608452338d9a5f12f58e8e7504c35763b03")
else()
  set(REQ_URL "https://github.com/symengine/symengine/archive/refs/tags/v0.14.0.tar.gz")
  set(SHA256 "11c5f64e9eec998152437f288b8429ec001168277d55f3f5f1df78e3cf129707")
endif()

akg_add_pkg(symengine
  VER 0.14.0
  LIBS symengine
  URL ${REQ_URL}
  SHA256 ${SHA256}
  CMAKE_OPTION -DHAVE_SYMENGINE_NOEXCEPT:BOOL=OFF -DWITH_BFD=OFF -DWITH_SYMENGINE_ASSERT:BOOL=OFF
    -DWITH_SYMENGINE_RCP:BOOL=ON -DWITH_SYMENGINE_THREAD_SAFE:BOOL=OFF -DWITH_SYMENGINE_THREAD_SAFE:BOOL=OFF
    -DWITH_ECM:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_BENCHMARKS:BOOL=OFF -DBUILD_BENCHMARKS_GOOGLE:BOOL=OFF
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
  CUSTOM_CMAKE_GENERATOR Ninja)

message(STATUS "symengine_inc :${symengine_INC}")
include_directories(${symengine_INC})
add_library(akg::symengine ALIAS symengine::symengine)
