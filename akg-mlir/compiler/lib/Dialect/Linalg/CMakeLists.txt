add_subdirectory(IR)
file(GLOB_RECURSE SRC_LIST *.cpp)

add_mlir_dialect_library(AKGLinalgPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Linalg
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Linglg/Transforms

  DEPENDS
  AKGLinalgPassIncGen
  MLIRLinalgDialect
  MLIRLinalgExtDialect
  MLIRMemRefDialect

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRLinalgExtDialect
  AKGMLIRAnalysis
  )
