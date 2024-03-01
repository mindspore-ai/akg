add_akg_translation_library(AkgTargetJSONToMLIR
  TranslateRegistration.cpp
  TranslateToMindsporeDialect.cpp
 
  ADDITIONAL_HEADER_DIRS
  ${AKG_SRC_INCLUDE_DIR}/akg/Target/MindsporeDialect
 
  LINK_COMPONENTS
  Core
 
  DISABLE_INSTALL
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationTransforms
  MLIRTosaDialect
  MLIRLinalgDialect
  AKGMindSporeToJson
  )