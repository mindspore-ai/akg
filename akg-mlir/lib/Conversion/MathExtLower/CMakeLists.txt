add_mlir_conversion_library(MathExtLower
  MathExtLower.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/MathExtLower
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Dialect/MindSpore
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVM

  DEPENDS
  AKGIRConversionPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MindSporeDialect
  MLIRMathExtDialect
)