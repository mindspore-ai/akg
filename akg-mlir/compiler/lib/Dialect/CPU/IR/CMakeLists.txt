add_mlir_dialect_library(MLIRCPUDialect
  CPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/CPU
  DISABLE_INSTALL
  DEPENDS
  MLIRCPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRLLVMDialect
  )