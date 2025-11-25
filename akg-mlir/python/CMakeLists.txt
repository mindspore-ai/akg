set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(AKG_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python_packages")

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=akg_mlir.")

declare_mlir_python_sources(AkgMLIRPythonSources)
declare_mlir_python_sources(AkgMLIRPythonExtensions)

set(_source_components
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
)

if(AKG_MLIR_ENABLE_PYAKG_EXTENSIONS)
  list(APPEND _source_components AkgMLIRPythonAkgExtensionsSources)
endif()

add_mlir_python_common_capi_library(AkgMLIRAggregateCAPI
  INSTALL_COMPONENT AkgMLIRPythonModules
  INSTALL_DESTINATION python_packages/akg/akg_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${AKG_MLIR_PYTHON_PACKAGES_DIR}/akg/akg_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(AkgMLIRPythonModules
  ROOT_PREFIX "${AKG_MLIR_PYTHON_PACKAGES_DIR}/akg/akg_mlir"
  INSTALL_PREFIX "python_packages/akg/akg_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    AkgMLIRAggregateCAPI
)
