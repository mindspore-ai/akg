file(GLOB_RECURSE SRC_LIST *.cpp)

add_mlir_dialect_library(MLIRFusionDialect
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Fusion

  DEPENDS
  MLIRFusionOpsIncGen

  DISABLE_INSTALL
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRVectorDialect
  MLIRMemRefDialect
)