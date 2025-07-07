include(FetchContent)
FetchContent_Declare(json URL ${JSON_TARBALL_URL})
FetchContent_MakeAvailable(json)

add_mlir_dialect_library(AKGTransformsPasses
  CopyRemoval.cpp
  CopyElision.cpp
  InferSymbolicShapes.cpp
  RemoveSymbolic.cpp
  SymbolRemoval.cpp
  StoreLoadElim.cpp
  PromoteTempBuffer.cpp
  CopyAttributesToGpu.cpp
  DumpShapeInfo.cpp
  LoadGlobalConfig.cpp
  AKGFuncOutlining.cpp
  AKGParallelLaunch.cpp
  AKGOperatorIdentify.cpp
  FoldDimension.cpp

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
