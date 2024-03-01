add_mlir_conversion_library(MindSporeToTosa
  MindSporeToTosa.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/MindSporeToTosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MindSpore
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  AKGIRConversionPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MindSporeDialect
)