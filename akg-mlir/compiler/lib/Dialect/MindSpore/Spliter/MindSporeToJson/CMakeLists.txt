# Integrate json lib
include(FetchContent)
FetchContent_Declare(json URL ${JSON_TARBALL_URL})
FetchContent_MakeAvailable(json)

add_mlir_library(AKGMindSporeToJson
  MindSporeToJson.cpp  
  
  DEPENDS
  MLIRIR
  
  LINK_LIBS PUBLIC
  MLIRIR
  nlohmann_json::nlohmann_json
)
