set(AKG_MLIR_LIBS
  AKGAffinePasses
  AKGTransformsPasses
  AKGGPUPasses
  AKGLinalgPasses
  AKGLLVMIRPasses
  AKGLinalgExtDialect
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

add_mlir_library(AKGGPUPipelines
  GPUOpt.cpp
  GPUDynOpt.cpp
  GPUCodegen.cpp

  DEPENDS
  ${AKG_MLIR_LIBS}
  DISABLE_INSTALL

  LINK_LIBS PUBLIC
  ${LIBS}
)
