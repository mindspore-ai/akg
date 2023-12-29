file(GLOB_RECURSE SRC_LIST *.cpp)

add_mlir_dialect_library(AKGAffinePasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Affine
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Affine/Transforms
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Affine/Analysis
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Affine/Utils

  DEPENDS
  AKGAffinePassIncGen
  MLIRAffineDialect
  MLIRMemRefDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  )
