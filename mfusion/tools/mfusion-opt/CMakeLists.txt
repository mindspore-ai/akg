get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRMfuseDialect
  MLIRDvmDialect
)

add_llvm_executable(mfusion-opt
  mfusion-opt.cc

  DEPENDS
  MLIRMfuseDialect
  MLIRDvmDialect
  MFusionConversionPassIncGen
)

# Add Torch-MLIR libraries (compiled by Torch-MLIR)
list(APPEND LIBS
  TorchMLIRInitAll
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRTMTensorDialect
)

llvm_update_compile_flags(mfusion-opt)
target_link_libraries(mfusion-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(mfusion-opt)
