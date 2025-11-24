file(GLOB_RECURSE SRC_LIST *.cpp)
FetchContent_Declare(json URL ${JSON_TARBALL_URL})
FetchContent_MakeAvailable(json)

add_mlir_dialect_library(AKGGPUPasses
  ${SRC_LIST}

  ADDITIONAL_HEADER_DIRS
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/GPU
  ${AKG_MLIR_SOURCE_DIR}/akg/Dialect/GPU/Transforms

  LINK_COMPONENTS
  Core
  MC
  ${NVPTX_LIBS}
  ${AMDGPU_LIBS}

  DISABLE_INSTALL
  DEPENDS
  AKGGPUPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRDataLayoutInterfaces
  MLIRExecutionEngineUtils
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils

  nlohmann_json::nlohmann_json
  )