file(GLOB_RECURSE SRC_LIST *.cpp)

add_mlir_dialect_library(AKGLLVMIRPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/LLVMIR
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/LLVMIR/Transforms

  DEPENDS
  AKGLLVMIRPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  )