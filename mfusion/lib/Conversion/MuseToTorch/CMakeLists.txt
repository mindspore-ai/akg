add_mlir_pdll_library(MuseToTorchPDLLIncGen
  MuseToTorch.pdll
  MuseToTorch.pdll.h.inc

  DEPENDS
  MuseMetaToTorch.pdll
  MuseAclnnToTorch.pdll
)

file(GLOB MUSE_TO_TORCH_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MFusionConversionMuseToTorch
  ${MUSE_TO_TORCH_SRC}

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
  MFusionConversionPassIncGen
  MuseToTorchPDLLIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRMuseDialect
  TorchMLIRTorchDialect
)

# Add generated include directory
target_include_directories(MFusionConversionMuseToTorch PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
