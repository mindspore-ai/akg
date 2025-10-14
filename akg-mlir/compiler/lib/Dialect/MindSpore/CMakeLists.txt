find_library(GMP_LIBRARY NAMES gmp REQUIRED)
add_subdirectory(IR)

file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(MindSporePasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/MindSpore
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Tosa/Transforms

  DEPENDS
  MindSporePassIncGen
  MindSporeDialect
  MLIRTosaDialect
  MLIRMemRefDialect

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MindSporeDialect
  MLIRFuncDialect
  MLIRTosaDialect
  MLIRTransformUtils
  AKGMLIRAnalysis
  )
target_link_libraries(MindSporePasses PRIVATE akg::symengine ${GMP_LIBRARY} stdc++fs)
