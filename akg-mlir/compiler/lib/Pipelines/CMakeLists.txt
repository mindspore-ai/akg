FetchContent_Declare(json URL ${JSON_TARBALL_URL})
FetchContent_MakeAvailable(json)
add_subdirectory(GPUPipelines)
set(AKG_MLIR_LIBS
  AKGTosaPasses
  MindSporePasses
  AKGAffinePasses
  AKGTransformsPasses
  AKGGPUPasses
  AKGSCFPasses
  AKGLinalgPasses
  AKGLLVMIRPasses
  AKGCPUPasses
  MindSporePasses
  MLIRLinalgExtDialect
  AKGMLIRAnalysis
  nlohmann_json::nlohmann_json
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

add_mlir_library(AkgPipelines
  CPUOpt.cpp
  CommonOpt.cpp

  DEPENDS
  ${AKG_MLIR_LIBS}
  AKGGPUPipelines
  
  LINK_LIBS PUBLIC
  ${LIBS}
  AKGGPUPipelines
  )
