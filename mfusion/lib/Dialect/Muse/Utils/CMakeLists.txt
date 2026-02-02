file(GLOB MUSE_UTILS_SRC "*.cc")

add_mlir_dialect_library(MLIRMuseUtils
  ${MUSE_UTILS_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mfusion

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSupport
)

