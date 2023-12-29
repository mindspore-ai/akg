add_mlir_dialect_library(MLIRLinalgExtDialect
  LinalgExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Linalg

  DEPENDS
  MLIRLinalgExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRLinalgUtils
  MLIRLinalgDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFusionDialect
)