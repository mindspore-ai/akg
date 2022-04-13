option(AKG_USE_MLS "Use MLSched polyhedral scheduler" ON)

if (AKG_USE_MLS)
  # Check for lfs files
  execute_process(
    COMMAND
      bash "${AKG_SOURCE_DIR}/build.sh" -o
    WORKING_DIRECTORY
      "${AKG_SOURCE_DIR}"
    OUTPUT_VARIABLE
      AKG_MLS_LFS_OUTPUT
    RESULT_VARIABLE
      AKG_MLS_LFS_RESULT
  )

  if (AKG_MLS_LFS_RESULT EQUAL 0)
    execute_process(
      COMMAND
        uname -m
      OUTPUT_VARIABLE
        AKG_MLS_ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (DEFINED AKG_GLIBCXX_USE_CXX11_ABI AND NOT AKG_GLIBCXX_USE_CXX11_ABI)
      set(AKG_MLS_CXX11 "old-abi.")
    endif ()

    add_library(libqiuqi-ip STATIC IMPORTED)
    set_target_properties(libqiuqi-ip
      PROPERTIES
        IMPORTED_LOCATION
          "${PROJECT_SOURCE_DIR}/prebuild/${AKG_MLS_ARCH}/libqiuqi_ip.a"
    )

    add_library(libmls STATIC IMPORTED)
    set_target_properties(libmls
      PROPERTIES
        IMPORTED_LOCATION
          "${PROJECT_SOURCE_DIR}/prebuild/${AKG_MLS_ARCH}/libmls.${AKG_MLS_CXX11}a"
        INTERFACE_INCLUDE_DIRECTORIES
          "${PROJECT_SOURCE_DIR}/src/"
    )

    add_compile_definitions(AKG_USE_MLS)
    target_link_libraries(akg libmls)
    target_link_libraries(akg libqiuqi-ip)
  else ()
    message(WARNING "${AKG_MLS_LFS_OUTPUT}")
  endif ()
endif()
