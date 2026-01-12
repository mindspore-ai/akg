file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGTensorPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Tensor
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Tensor/Transforms

  DEPENDS
  AKGTensorPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRTransformUtils
  )
