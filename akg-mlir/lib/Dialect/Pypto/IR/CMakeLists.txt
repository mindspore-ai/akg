add_mlir_dialect_library(AKGPyptoDialect
  PyptoDialect.cpp
  PyptoOps.cpp
  PyptoAttrs.cpp
  PyptoTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/include/akg/Dialect/Pypto/IR

  DISABLE_INSTALL

  DEPENDS
  MLIRPyptoOpsIncGen
  AKGPyptoAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
