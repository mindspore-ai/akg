add_mlir_dialect_library(MLIRMathExtDialect
  MathExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Math

  DEPENDS
  MLIRMathExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRLLVMDialect
  )