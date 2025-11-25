link_directories(${AKG_BIN_LIB_DIR})
add_executable(akg-ascend-linker ascend-linker.cpp)
target_link_libraries(akg-ascend-linker PRIVATE akg_ascend_launch_runtime)