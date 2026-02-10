file(GLOB MFUSION_CAPI_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_public_c_api_library(MFusionCAPI
  ${MFUSION_CAPI_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mfusion-c/

  DEPENDS
  MFusionConversionPassIncGen
  MLIRMfuseTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRMfuseDialect
  MFusionConversionTorchToMfuse
  MFusionConversionMfuseToTorch
  MLIRMfuseTransforms
)

# Add include directories
target_include_directories(MFusionCAPI PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
