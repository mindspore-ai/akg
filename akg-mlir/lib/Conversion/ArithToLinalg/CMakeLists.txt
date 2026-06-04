add_mlir_conversion_library(MLIRArithToLinalg
  ArithToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/ArithToLinalg

  DISABLE_INSTALL

  DEPENDS
  AKGIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRRewrite
  MLIRPass
  )