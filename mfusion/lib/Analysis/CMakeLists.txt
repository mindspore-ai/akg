file(GLOB_RECURSE MFUSION_ANALYSIS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_library(MFusionAnalysis
  ${MFUSION_ANALYSIS_SRC}


  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  mfusion::symengine
)
