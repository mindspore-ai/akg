add_mlir_library(AKGMLIRAnalysis
  AutoTiling.cpp
  Axis.cpp
  BufferAnalysis.cpp
  Config.cpp
  LoopTiling.cpp
  Model.cpp
  SymbolicShapeAnalysis.cpp
  TilingSolver.cpp
  TilingStrategy.cpp
  TypeUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Analysis
  ${SYMENGINE_INCLUDE_DIRS}/

  DEPENDS
  AKGAffinePasses

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  AKGAffinePasses
  BiShengIRHACCDialect
  MLIRAffineDialect
  MLIRAnalysis
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRShapeDialect
  MLIRTensorDialect
)


