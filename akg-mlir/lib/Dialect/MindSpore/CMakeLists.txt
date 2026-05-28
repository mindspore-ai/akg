find_library(GMP_LIBRARY NAMES gmp REQUIRED)
add_subdirectory(IR)

file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGMindSporePasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/MindSpore
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Tosa/Transforms

  DEPENDS
  AKGMindSporePassIncGen
  AKGMindSporeDialect
  MLIRTosaDialect
  MLIRMemRefDialect

  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRTosaDialect
  MLIRTransformUtils
  AKGMindSporeDialect
  AKGMLIRAnalysis
  )
target_link_libraries(AKGMindSporePasses PRIVATE akg::symengine ${GMP_LIBRARY} stdc++fs)
