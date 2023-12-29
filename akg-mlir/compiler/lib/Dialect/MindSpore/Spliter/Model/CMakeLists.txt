add_mlir_library(AKGModel
  Area.cpp
  FusePattern.cpp
  Model.cpp
  CpuModel.cpp
  AscendModel.cpp
  GpuModel.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SRC_INCLUDE_DIR}/akg/Dialect/MindSpore
  
  DEPENDS
  AKGMindSporeToJson
  AKGLiteIR

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRSupport
  MLIRIR
  AKGMindSporeToJson
  AKGLiteIR
  )
