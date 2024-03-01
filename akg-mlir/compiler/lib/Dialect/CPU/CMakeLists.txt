add_subdirectory(IR)

file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGCPUPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/CPU

  DEPENDS
  AKGCPUPassIncGen
  MLIRCPUDialect
  
  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRCPUDialect
  MLIRTransformUtils
  AKGMLIRAnalysis
  )
target_link_libraries(AKGCPUPasses PRIVATE akg::symengine gmp stdc++fs)
