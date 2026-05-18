
add_mlir_pdll_library(TorchToMfusePDLLIncGen
  TorchToMfuse.pdll
  TorchToMfuse.pdll.h.inc

  DEPENDS
  TorchAtenToMfuse.pdll
  TorchNpuToMfuse.pdll
  TorchPrimToMfuse.pdll
)

file(GLOB TORCH_TO_MFUSE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MFusionConversionTorchToMfuse
  ${TORCH_TO_MFUSE_SRC}

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
  MFusionConversionPassIncGen
  TorchToMfusePDLLIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSupport
  MLIRTransformUtils
  MLIRMfuseDialect
  MFusionAnalysis
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRParser
)

# Add generated include directory
target_include_directories(MFusionConversionTorchToMfuse PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
