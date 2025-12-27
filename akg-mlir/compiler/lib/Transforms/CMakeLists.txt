include(FetchContent)
FetchContent_Declare(json URL ${JSON_TARBALL_URL})
FetchContent_MakeAvailable(json)
file(GLOB_RECURSE SRC_LIST *.cpp)

add_mlir_dialect_library(AKGTransformsPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Transforms
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/CPU

  DISABLE_INSTALL
  DEPENDS
  AKGTransformsPassIncGen
  MLIRCPUDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRCPUDialect

  nlohmann_json::nlohmann_json
  )
