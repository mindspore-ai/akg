file(GLOB MFUSE_TO_DVM_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MFusionConversionMfuseToDvm
  ${MFUSE_TO_DVM_SRC}

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
  MFusionConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSupport
  MLIRTransforms
  MLIRMfuseDialect
  MLIRDvmDialect
)
