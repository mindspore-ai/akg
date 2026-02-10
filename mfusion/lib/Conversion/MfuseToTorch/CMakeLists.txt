add_mlir_pdll_library(MfuseToTorchPDLLIncGen
  MfuseToTorch.pdll
  MfuseToTorch.pdll.h.inc

  DEPENDS
  MfuseMetaToTorch.pdll
  MfuseAclnnToTorch.pdll
)

file(GLOB MFUSE_TO_TORCH_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MFusionConversionMfuseToTorch
  ${MFUSE_TO_TORCH_SRC}

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
  MFusionConversionPassIncGen
  MfuseToTorchPDLLIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRMfuseDialect
  TorchMLIRTorchDialect
)

# Add generated include directory
target_include_directories(MFusionConversionMfuseToTorch PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
