# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.
find_package(OpenMP REQUIRED)
append("-fopenmp" CMAKE_CXX_FLAGS)
set(LLVM_OPTIONAL_SOURCES
AKGParallelLaunchRuntime.cpp
)
# Use a separate library for OptUtils, to avoid pulling in the entire JIT and
# codegen infrastructure. Unlike MLIRExecutionEngine, this is part of
# libMLIR.so.
add_library(mlir_akgParallelLaunch_runtime
SHARED
AKGParallelLaunchRuntime.cpp
)
set_property(TARGET mlir_akgParallelLaunch_runtime PROPERTY CXX_STANDARD 17)
target_compile_definitions(mlir_akgParallelLaunch_runtime PRIVATE mlir_akgParallelLaunch_runtime_EXPORTS)