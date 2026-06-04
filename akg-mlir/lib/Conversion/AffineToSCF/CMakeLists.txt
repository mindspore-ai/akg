add_mlir_conversion_library(AKGMLIRAffineToSCF
  AffineToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/AffineToSCF

  DEPENDS
  AKGIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRAffineUtils
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )

