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
  MLIRAffineAnalysis
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  MLIRCPUDialect

  akg::nlohmann_json
  )
