include_directories(${GTEST_DIR}/googletest/include)
include_directories(${UNITTEST_CPP_SRC}include)
include_directories(${AKG_SOURCE_DIR}/src)
include_directories(${TVM_DIR}/include)
include_directories(${TVM_DIR}/topi/include)

file(
  GLOB
  UT_CPP_SRC
  unittest_main.cc
  src/base/*.cc
  src/base_test/*.cc
  src/pass_test/*.cc)

add_executable(unittest_main ${UT_CPP_SRC})

target_link_libraries(unittest_main gtest akg ${TVM_RUNTIME_LINKER_LIBS} ${CMAKE_DL_LIBS})
