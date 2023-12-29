file(GLOB_RECURSE SRC_LIST *.cpp)
add_mlir_dialect_library(AKGTosaPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Tosa
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/Tosa/Transforms

  DEPENDS
  AKGTosaPassIncGen
  MLIRTosaDialect
  MLIRMemRefDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRTosaDialect
  MLIRTransformUtils
  AKGMLIRAnalysis
  )
target_link_libraries(AKGTosaPasses PRIVATE ${SYMENGINE_LIBRARIES} stdc++fs)
