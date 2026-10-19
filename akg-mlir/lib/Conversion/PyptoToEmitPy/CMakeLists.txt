add_mlir_conversion_library(AKGPyptoToEmitPy
  PyptoToEmitPy.cpp

  DEPENDS
  AKGIRConversionPassIncGen

  DISABLE_INSTALL

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransformUtils
  AKGEmitPyDialect
  AKGPyptoDialect
)
