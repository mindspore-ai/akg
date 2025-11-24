add_mlir_conversion_library(MindSporeToLinalg
  MindSporeToLinalg.cpp
  MindSporeToLinalgNamed.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/MindSporeToLinalg
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Dialect/MindSpore
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Dialect/Math

  DEPENDS
  AKGIRConversionPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MindSporeDialect
)