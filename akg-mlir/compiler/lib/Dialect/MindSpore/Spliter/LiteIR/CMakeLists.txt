add_mlir_library(AKGLiteIR
  Node.cpp
  OpRegister.cpp
  LiteGraph.cpp
  OpNode.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SRC_INCLUDE_DIR}/akg/Dialect/MindSpore
  
  DEPENDS
  AKGMindSporeToJson

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRSupport
  MLIRIR
  AKGMindSporeToJson
  )
