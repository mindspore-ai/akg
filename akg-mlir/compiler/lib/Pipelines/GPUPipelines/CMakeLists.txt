set(AKG_MLIR_LIBS
  AKGTosaPasses
  AKGAffinePasses
  AKGTransformsPasses
  AKGGPUPasses
  AKGLinalgPasses
  AKGLLVMIRPasses
  MLIRLinalgExtDialect
  AKGMLIRAnalysis
  libpolytops-schedule-opt
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

  LINK_LIBS PUBLIC
  ${LIBS}
)
