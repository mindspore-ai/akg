option(AKG_USE_MLS "Use MLSched polyhedral scheduler" OFF)

if(NOT USE_CUDA AND NOT USE_LLVM)
  set(AKG_USE_MLS ON)
endif()

if (AKG_USE_MLS)
  execute_process(
    COMMAND
      uname -m
    OUTPUT_VARIABLE
      AKG_MLS_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  add_custom_target(akg-untar-mls
    ALL
    COMMAND
      "${CMAKE_COMMAND}" -E tar xfz ${AKG_SOURCE_DIR}/prebuild/${AKG_MLS_ARCH}/libmls.tar.gz
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/"
    DEPENDS
      "${AKG_SOURCE_DIR}/prebuild/${AKG_MLS_ARCH}/libmls.tar.gz"
  )

  add_library(libqiuqi-ip STATIC IMPORTED)
  set_target_properties(libqiuqi-ip
    PROPERTIES
      IMPORTED_LOCATION
        "${CMAKE_CURRENT_BINARY_DIR}/libqiuqi_ip.a"
    DEPENDS
      akg-untar-mls
  )

  add_library(libmls STATIC IMPORTED)
  set_target_properties(libmls
    PROPERTIES
      IMPORTED_LOCATION
        "${CMAKE_CURRENT_BINARY_DIR}/libmls.a"
      INTERFACE_INCLUDE_DIRECTORIES
        "${AKG_SOURCE_DIR}/src/poly"
    DEPENDS
      akg-untar-mls
  )

  add_compile_definitions(AKG_USE_MLS)
  target_link_libraries(akg libmls)
  target_link_libraries(akg libqiuqi-ip)
endif()
