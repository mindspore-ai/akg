add_subdirectory(GPUPipelines)
add_subdirectory(AscendPipelines)
set(AKG_MLIR_LIBS
  AKGMindSporePasses
  AKGAffinePasses
  AKGTransformsPasses
  AKGGPUPasses
  AKGSCFPasses
  AKGLinalgPasses
  AKGLLVMIRPasses
  AKGCPUPasses
  AKGLinalgExtDialect
  AKGMLIRAnalysis
  akg::nlohmann_json
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

add_mlir_library(AKGPipelines
  CPUOpt.cpp
  CommonOpt.cpp

  DEPENDS
  ${AKG_MLIR_LIBS}
  AKGGPUPipelines
  AKGAscendPipelines

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  ${LIBS}
  AKGGPUPipelines
  AKGAscendPipelines
  )
