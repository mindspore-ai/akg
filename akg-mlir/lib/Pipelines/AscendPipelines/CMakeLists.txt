set(AKG_MLIR_LIBS
  AKGTransformsPasses
  AKGLinalgPasses
  AKGTensorPasses
  AKGVectorPasses
  AKGLLVMIRPasses
  AKGMLIRAnalysis
  AKGNPUVectorPasses
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

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(AKGAscendPipelines
  AscendOpt.cpp

  DEPENDS
  ${AKG_MLIR_LIBS}
  ${conversion_libs}

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  ${LIBS}
  ${conversion_libs}

)
