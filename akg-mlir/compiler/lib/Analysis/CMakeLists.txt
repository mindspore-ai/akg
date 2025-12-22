add_mlir_library(AKGMLIRAnalysis
  BufferAnalysis.cpp
  SymbolicShapeAnalysis.cpp
  TypeUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Analysis
  ${SYMENGINE_INCLUDE_DIRS}/

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRShapeDialect
  MLIRTensorDialect
)


