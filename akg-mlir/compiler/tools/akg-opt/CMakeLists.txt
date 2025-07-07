set(AKG_MLIR_LIBS
  AKGAffinePasses
  AKGTransformsPasses
  AKGSCFPasses
  AKGGPUPasses
  AKGLinalgPasses
  AkgPipelines
  MLIRLinalgExtDialect
  MLIRFusionDialect
  AKGMLIRAnalysis
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
  AKGLinalgPasses
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRMlirOptMain
  )

link_directories(${AKG_MLIR_BIN_LIB_DIR})

add_executable(akg-opt akg-opt.cpp)
add_dependencies(akg-opt ${LIBS})

llvm_update_compile_flags(akg-opt)
target_link_libraries(akg-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(akg-opt)
