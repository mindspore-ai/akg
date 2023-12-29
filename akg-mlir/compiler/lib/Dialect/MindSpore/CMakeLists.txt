add_subdirectory(IR)
add_subdirectory(Spliter)

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
  AKGMindSporeToJson
  AKGSpliter
  )
target_link_libraries(MindSporePasses PRIVATE ${SYMENGINE_LIBRARIES} stdc++fs)