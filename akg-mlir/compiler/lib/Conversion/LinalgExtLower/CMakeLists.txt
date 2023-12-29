add_mlir_conversion_library(MLIRLinalgExtLower
  LinalgExtLower.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/LinalgExtLower
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  AKGIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MLIRLinalgExtDialect
)