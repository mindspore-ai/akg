# Unlike mlir_float16_utils, mlir_c_runner_utils, etc, we do *not* make
# this a shared library: because doing so causes issues when building
# on Windows.  In particular, various functions take/return `std::vector`
# but that class is not designed with dllimport/dllexport pragma,
# therefore it cannot be safely/correctly used across DLL boundaries.
# Consequently, we avoid using the "mlir_xxx_utils" naming scheme, since
# that is reserved/intended for shared libraries only.
add_library(ascend_launch_runtime
  SHARED
  CceWrapper.cpp
  AscendLaunchRuntime.cpp
  AscendMemoryManager.cpp
  logger.cpp
  RuntimeErrorCodes.cpp
)
set_property(TARGET ascend_launch_runtime PROPERTY CXX_STANDARD 17)
target_link_libraries(ascend_launch_runtime glog dl akg::nlohmann_json)

check_cxx_compiler_flag(-Wweak-vtables COMPILER_SUPPORTS_WARNING_WEAK_VTABLES)
if(COMPILER_SUPPORTS_WARNING_WEAK_VTABLES)
  target_compile_options(ascend_launch_runtime PUBLIC "-Wweak-vtables")
endif()

pybind11_add_module(ascend_launch AscendLaunch.cpp)
target_link_libraries(ascend_launch PRIVATE ascend_launch_runtime)
set_target_properties(ascend_launch PROPERTIES BUILD_RPATH $ORIGIN:$ORIGIN/lib)
set_target_properties(ascend_launch PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/lib)
pybind11_add_module(profile_mgr ProfileMgr.cpp)
target_link_libraries(profile_mgr PRIVATE ascend_launch_runtime)
set_target_properties(profile_mgr PROPERTIES BUILD_RPATH $ORIGIN:$ORIGIN/lib)
set_target_properties(profile_mgr PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/lib)
