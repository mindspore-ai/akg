file(GLOB MUSE_DIALECT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_dialect_library(MLIRMuseDialect
  ${MUSE_DIALECT_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mfusion

  DEPENDS
  MLIRMuseIncGen
  MLIRMuseExtraIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_subdirectory(Transforms)
