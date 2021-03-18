add_compile_options(-std=c++11)
include_directories(${GTEST_DIR}/googletest/include)
include_directories(${UNITTEST_DIR}include)
include_directories(${UNITTEST_DIR}src)

include_directories(${AKG_SOURCE_DIR}/src)
include_directories(${AKG_SOURCE_DIR}/src/include)

include_directories(${TVM_DIR}/include)
include_directories(${TVM_DIR}/src)
include_directories(${TVM_DIR}/topi/include)
include_directories(AFTER "${TVM_DIR}/3rdparty/dmlc-core/include")
include_directories(AFTER "${TVM_DIR}/3rdparty/dlpack/include")

include_directories(${AKG_SOURCE_DIR}/third_party/isl_wrap/include)
include_directories(${ISL_DIR}/include)
include_directories(${ISL_DIR})

file(
  GLOB
  UT_CPP_SRC
  unittest_main.cc
  src/base/*.cc
  src/base_test/*.cc
  src/pass_test_base/*.cc
  src/pass_test/*.cc
  src/poly_pass_test/*.cc)

link_directories(${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)

add_executable(unittest_main ${UT_CPP_SRC})

target_link_libraries(unittest_main PRIVATE akg akg::gtest ${TVM_RUNTIME_LINKER_LIBS} rt dl pthread)
