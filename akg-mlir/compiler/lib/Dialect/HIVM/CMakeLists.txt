file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGHIVMPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/HIVM
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/HIVM/Transforms

  DEPENDS
  AKGHIVMPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTransformUtils
  )
