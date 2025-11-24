add_mlir_conversion_library(MLIRTosaToLinalgUpdate
  TosaMultiReduceToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/TosaToLinalgUpdate
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  AKGIRConversionPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MLIRTosaDialect
  MLIRLinalgDialect
)