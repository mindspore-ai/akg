file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGVectorPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Vector
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Vector/Transforms

  DEPENDS
  AKGVectorPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRTransformUtils
  )
