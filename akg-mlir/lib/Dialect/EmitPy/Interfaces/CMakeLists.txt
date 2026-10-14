add_mlir_library(AKGEmitPyInterfaces
  EmitPyInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/include/akg/Dialect/EmitPy/Interfaces

  DISABLE_INSTALL

  DEPENDS
  MLIREmitPyInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
