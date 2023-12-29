add_mlir_conversion_library(
  MLIRPureOpenMPToLLVM PureOpenMPToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/PureOpenMPToLLVM

  DEPENDS
  AKGIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MLIROpenMPDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)