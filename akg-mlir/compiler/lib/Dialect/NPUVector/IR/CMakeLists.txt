add_mlir_dialect_library(NPUVectorDialect
  NPUVectorDialect.cpp
  NPUVectorOps.cpp
  NPUVectorTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/include/akg/Dialect/NPUVector/IR

  DISABLE_INSTALL

  DEPENDS
  MLIRNPUVectorOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRVectorUtils
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  MLIRDestinationStyleOpInterface
  MLIRVectorInterfaces
)

