add_subdirectory(LiteIR)
add_subdirectory(MindSporeToJson)
add_subdirectory(Model)
add_mlir_library(AKGSpliter
  Spliter.cpp 
  LINK_LIBS PUBLIC
  AKGLiteIR
  AKGMindSporeToJson
  AKGModel
)
