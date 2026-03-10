file(GLOB MFUSE_DIALECT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_dialect_library(MLIRMfuseDialect
  ${MFUSE_DIALECT_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mfusion

  DEPENDS
  MLIRMfuseIncGen
  MLIRMfuseExtraIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRMfuseUtils
  mfusion::symengine
)
