add_mlir_conversion_library(AKGLinalgToPypto
  LinalgToPypto.cpp

  DEPENDS
  AKGIRConversionPassIncGen

  DISABLE_INSTALL

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTransformUtils
  AKGPyptoDialect
)
