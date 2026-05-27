add_subdirectory(IR)

file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGNPUVectorPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/NPUVector

  DEPENDS
  AKGNPUVectorPassIncGen
  NPUVectorDialect
  MLIRMemRefDialect

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  NPUVectorDialect
  MLIRFuncDialect
  MLIRTransformUtils
  AKGMLIRAnalysis
  )

