add_mlir_conversion_library(MindSporeFinalizingLower
  MindSporeFinalizingLower.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/MindSporeFinalizingLower
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MindSpore
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
  DISABLE_INSTALL
  
  DEPENDS
  AKGIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MindSporeDialect
)