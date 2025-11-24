add_mlir_conversion_library(MLIRFuncToLLVMExt
  FuncToLLVMExt.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/akg/Conversion/FuncToLLVMExt

  DEPENDS
  AKGIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransformUtils
  )