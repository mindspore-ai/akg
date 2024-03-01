file(GLOB_RECURSE SRC_LIST *.cpp)

add_mlir_dialect_library(AKGSCFPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/SCF
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/SCF/Transforms

  DEPENDS
  AKGSCFPassIncGen
  MLIRSCFDialect
  
  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRDataLayoutInterfaces
  MLIRExecutionEngineUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  )