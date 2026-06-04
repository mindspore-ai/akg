# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.
find_package(OpenMP REQUIRED)

set(LLVM_OPTIONAL_SOURCES
  ParallelLaunchRuntime.cpp
  AscendLaunchRuntime.cpp
)
# Use a separate library for OptUtils, to avoid pulling in the entire JIT and
# codegen infrastructure. Unlike MLIRExecutionEngine, this is part of
# libMLIR.so.
add_library(cpu_launch_runtime
  SHARED
  ParallelLaunchRuntime.cpp
)
set_property(TARGET cpu_launch_runtime PROPERTY CXX_STANDARD 17)
target_link_libraries(cpu_launch_runtime PRIVATE OpenMP::OpenMP_CXX)

#set AscendLaunchRuntime
add_subdirectory(AscendLaunchRuntime)
