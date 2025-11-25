set(AKG_MLIR_LIBS
  AKGTransformsPasses
  AKGLinalgPasses
  AKGLLVMIRPasses
  AKGMLIRAnalysis
  )

set(LIBS
  ${AKG_MLIR_LIBS}
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