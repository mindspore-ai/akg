option(AKG_USE_POLYTOPS "Use PolyTOPS polyhedral scheduler" OFF)

if (NOT USE_CUDA AND NOT USE_LLVM)
  set(AKG_USE_POLYTOPS ON)
endif()

if (AKG_USE_POLYTOPS)
  execute_process(
    COMMAND
      uname -m
    OUTPUT_VARIABLE
      AKG_POLYTOPS_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  add_custom_target(akg-untar-polytops
    ALL
    COMMAND
      "${CMAKE_COMMAND}" -E tar xfz ${AKG_SOURCE_DIR}/prebuild/${AKG_POLYTOPS_ARCH}/libpolytops.tar.gz
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/"
    DEPENDS
      "${AKG_SOURCE_DIR}/prebuild/${AKG_POLYTOPS_ARCH}/libpolytops.tar.gz"
  )

  add_library(libqiuqi-ip STATIC IMPORTED)
  set_target_properties(libqiuqi-ip
    PROPERTIES
      IMPORTED_LOCATION
        "${CMAKE_CURRENT_BINARY_DIR}/libqiuqi_ip.a"
    DEPENDS
      akg-untar-polytops
  )

  add_library(libpolytops STATIC IMPORTED)
  set_target_properties(libpolytops
    PROPERTIES
      IMPORTED_LOCATION
        "${CMAKE_CURRENT_BINARY_DIR}/libpolytops.a"
      INTERFACE_INCLUDE_DIRECTORIES
        "${AKG_SOURCE_DIR}/src/poly"
    DEPENDS
      akg-untar-polytops
  )

  add_compile_definitions(AKG_USE_POLYTOPS)
  target_link_libraries(akg libpolytops)
  target_link_libraries(akg libqiuqi-ip)
endif()
