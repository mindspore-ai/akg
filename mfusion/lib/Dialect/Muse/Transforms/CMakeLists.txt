file(GLOB MUSE_TRANSFORMS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_dialect_library(MLIRMuseTransforms
  ${MUSE_TRANSFORMS_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mfusion

  DEPENDS
  MLIRMuseTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRAsmParser
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRMuseDialect
)
