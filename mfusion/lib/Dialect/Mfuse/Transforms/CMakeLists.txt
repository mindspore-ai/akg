file(GLOB_RECURSE MFUSE_TRANSFORMS_SRC "*.cc")

add_mlir_dialect_library(MLIRMfuseTransforms
  ${MFUSE_TRANSFORMS_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mfusion

  DEPENDS
  MLIRMfuseTransformsPassIncGen

  LINK_LIBS PUBLIC
  MFusionAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRAsmParser
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRRewrite
  MLIRMfuseDialect
  MLIRMfuseUtils
)
