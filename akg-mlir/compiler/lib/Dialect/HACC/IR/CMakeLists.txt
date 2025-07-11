add_mlir_dialect_library(HACCDialect
  HACCOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/HACC

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRLLVMDialect
  )