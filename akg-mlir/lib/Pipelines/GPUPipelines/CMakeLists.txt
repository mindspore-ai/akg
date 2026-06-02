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

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_mlir_library(AKGGPUPipelines
  GPUOpt.cpp
  GPUDynOpt.cpp
  GPUCodegen.cpp

  DEPENDS
  ${AKG_MLIR_LIBS}
  ${conversion_libs}
  DISABLE_INSTALL

  LINK_LIBS PUBLIC
  ${LIBS}
  ${conversion_libs}
)
