add_mlir_library(AKGMLIRAnalysis
  SymbolicShapeAnalysis.cpp
  TypeUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Analysis
  ${SYMENGINE_INCLUDE_DIRS}/
  

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRBufferizationTransforms
  MLIRIR
  MLIRShapeDialect
  MLIRTensorDialect
)


