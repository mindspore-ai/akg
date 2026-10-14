add_mlir_dialect_library(AKGEmitPyDialect
  EmitPyDialect.cpp
  EmitPyOps.cpp
  EmitPyTypes.cpp
  EmitPyAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/include/akg/Dialect/EmitPy/IR

  DISABLE_INSTALL

  DEPENDS
  MLIREmitPyOpsIncGen
  AKGEmitPyAttrsIncGen

  LINK_LIBS PUBLIC
  AKGEmitPyInterfaces
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
