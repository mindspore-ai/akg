# Test-only MFusion CAPI source list.
# This file is included by mfusion/lib/CAPI/CMakeLists.txt when BUILD_TESTS=ON.

file(GLOB_RECURSE MFUSION_TEST_CAPI_SRC "${CMAKE_CURRENT_LIST_DIR}/*.cc")
