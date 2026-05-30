add_subdirectory(IR)

file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGMathPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Math

  DEPENDS
  AKGMathPassIncGen
  MLIRMathDialect
  MLIRMathExtDialect
  MLIRMemRefDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMathExtDialect
  MLIRTransformUtils
  AKGMLIRAnalysis
  )
target_link_libraries(AKGMathPasses PRIVATE ${SYMENGINE_LIBRARIES} stdc++fs)
