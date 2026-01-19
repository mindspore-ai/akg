
add_mlir_pdll_library(TorchToMusePDLLIncGen
  TorchToMuse.pdll
  TorchToMuse.pdll.h.inc

  DEPENDS
  TorchToMuseAten.pdll
  TorchToMuseConstant.pdll
  TorchToMuseNpu.pdll
)

file(GLOB TORCH_TO_MUSE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MFusionConversionTorchToMuse
  ${TORCH_TO_MUSE_SRC}

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/include/mfusion/Conversion/TorchToMuse

  DEPENDS
  MFusionConversionPassIncGen
  TorchToMusePDLLIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSupport
  MLIRTransformUtils
  MLIRMuseDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRParser
)

# Add generated include directory
target_include_directories(MFusionConversionTorchToMuse PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
