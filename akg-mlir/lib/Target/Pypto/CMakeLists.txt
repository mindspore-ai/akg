
add_akg_translation_library(AkgTargetPython
  TranslateToPyptoRegistration.cpp
  TranslateToPypto.cpp

  ADDITIONAL_HEADER_DIRS
  ${AKG_SRC_INCLUDE_DIR}/akg/Target/Pypto

  DISABLE_INSTALL

  LINK_LIBS PUBLIC
  AKGEmitPyDialect
  AKGNPUVectorDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRVectorDialect
)
