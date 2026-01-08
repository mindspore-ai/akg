file(GLOB COMMON_TO_MUSE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MFusionConversionStandardToMuse
  ${COMMON_TO_MUSE_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mfusion/Conversion/StandardToMuse

  DEPENDS
  MFusionConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRMuseDialect
)
