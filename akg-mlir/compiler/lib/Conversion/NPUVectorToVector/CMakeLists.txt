add_mlir_conversion_library(AKGNPUVectorToVector
  NPUVectorToVector.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/NPUVectorToVector

  DEPENDS
  AKGIRConversionPassIncGen

  DISABLE_INSTALL

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRVectorDialect
  NPUVectorDialect
)
