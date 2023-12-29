
add_akg_translation_library(AkgTargetPTX
  TranslateRegistration.cpp
  TranslateToPTX.cpp
  GPUKernelToPTX.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_SRC_INCLUDE_DIR}/akg/Target/PTX

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRAMDGPUDialect
  MLIRIndexDialect
  MLIRMLProgramDialect
  MLIRPass
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTransforms
  MLIRTransformUtils
  MLIRTranslateLib
  MLIRGPUTransformOps
  MLIRAffineTransformOps
  MLIRLinalgTransformOps
  MLIRMemRefTransformOps
  MLIRVectorTransformOps
  MLIRSCFTransformOps
  MLIRBufferizationTransformOps
  MLIRShapeOpsTransforms
  )
