set(AKG_MLIR_LIBS
  AKGTosaPasses
  AKGAffinePasses
  AKGTransformsPasses
  AKGSCFPasses
  AKGGPUPasses
  AKGTosaPasses
  AKGLinalgPasses
  AkgPipelines
  MLIRLinalgExtDialect
  MLIRFusionDialect
  AKGMLIRAnalysis
  libpolytops-schedule-opt
)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
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




link_directories(${AKG_MLIR_BIN_LIB_DIR})
add_executable(akg-opt akg-opt.cpp)
add_dependencies(akg-opt ${LIBS} akg::symengine)

llvm_update_compile_flags(akg-opt)
target_link_libraries(akg-opt PRIVATE ${LIBS} akg::symengine)

mlir_check_all_link_libraries(akg-opt)
