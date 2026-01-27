file(GLOB_RECURSE MUSE_TRANSFORMS_SRC "*.cc")

add_mlir_dialect_library(MLIRMuseTransforms
  ${MUSE_TRANSFORMS_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mfusion

  DEPENDS
  MLIRMuseTransformsPassIncGen

  LINK_LIBS PUBLIC
  MFusionAnalysis
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRAsmParser
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRRewrite
  MLIRMuseDialect
)
