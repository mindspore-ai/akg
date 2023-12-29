add_mlir_conversion_library(MindSporeToLinalg
  MindSporeToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/MindSporeToLinalg
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Dialect/MindSpore
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Dialect/Math

  DEPENDS
  AKGIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MindSporeDialect
  MLIRMathExtDialect
)