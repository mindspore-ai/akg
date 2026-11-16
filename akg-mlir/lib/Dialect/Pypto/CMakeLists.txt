add_subdirectory(IR)

add_mlir_dialect_library(AKGPyptoPasses
  Transforms/PyptoAlignShape.cpp

  DEPENDS
  AKGPyptoPassIncGen

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  AKGPyptoDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
