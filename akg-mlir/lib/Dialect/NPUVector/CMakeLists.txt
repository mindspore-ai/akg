add_subdirectory(IR)

file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGNPUVectorPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/NPUVector

  DEPENDS
  AKGNPUVectorPassIncGen
  AKGNPUVectorDialect
  MLIRMemRefDialect

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRTransformUtils
  AKGNPUVectorDialect
  AKGMLIRAnalysis
  )

