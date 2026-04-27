set(AKG_MLIR_LIBS
  AKGTransformsPasses
  AKGLinalgPasses
  AKGTensorPasses
  AKGLLVMIRPasses
  AKGMLIRAnalysis
  )

set(LIBS
  ${AKG_MLIR_LIBS}
  BiShengIRArithToAffine
  BiShengIRHIVMToStandard
  BiShengIRHIVMTransforms
  BiShengIRHACCTransforms
  BiShengIRSCFTransforms
  BiShengIRScopeTransforms
  BiShengIRAnnotationTransforms
  BiShengIRHIVMUtils
  BiShengIRHACCUtils
  BiShengIRSCFUtils
  BiShengIRDialectUtils
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRSymbolDialect
  BiShengIRMathExtDialect
  BiShengIRDialectUtils
  BiShengIRTensorDialect
  BiShengIRMemRefDialect
  BiShengIRHIVMDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

add_mlir_library(AKGAscendPipelines
  AscendOpt.cpp

  DEPENDS
  ${AKG_MLIR_LIBS}

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  ${LIBS}
)
