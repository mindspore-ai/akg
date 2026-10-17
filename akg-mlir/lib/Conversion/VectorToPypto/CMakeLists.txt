add_mlir_conversion_library(AKGVectorToEmitPy
  VectorToPypto.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/include/akg/Conversion/VectorToPypto

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
  MLIRTransformUtils
  MLIRVectorDialect
  AKGEmitPyDialect
  AKGNPUVectorDialect
  AKGPyptoDialect
)
