include(CheckSymbolExists)
check_symbol_exists(__aarch64__ "" __CHECK_AARCH64)
check_symbol_exists(__x86_64__ "" __CHECK_X86_64)

add_definitions(-DUSE_CCE_RT=1)

# Search AKG_EXTEND by order
set(AKG_EXTEND )
if(USE_CCE_RT OR USE_KC_AIR)
  if(NOT __CHECK_AARCH64 AND NOT __CHECK_X86_64)
    message(FATAL_ERROR "runtime only support aarch64 and x86_64")
  endif()

  set(AKG_EXTEND_FILE )
  set(LIB_PATH1 ${AKG_SOURCE_DIR}/libakg_ext.a)
  set(LIB_PATH2 ${CMAKE_CURRENT_BINARY_DIR}/libakg_ext.a)

  if(EXISTS ${LIB_PATH1})  # Search libakg_ext.a in akg_source_dir/
    set(AKG_EXTEND_FILE ${LIB_PATH1})
    message("-- Find ${LIB_PATH1}")
  else()
    if(EXISTS ${LIB_PATH2})  # If .a not found, search .a in akg_build_dir/
      set(AKG_EXTEND_FILE ${LIB_PATH2})
      message("-- Find ${LIB_PATH2}")
    elseif(NOT USE_KC_AIR)  # If .a not found, search .o in akg_source_dir/prebuild
      execute_process(COMMAND bash ${AKG_SOURCE_DIR}/build.sh -o
              WORKING_DIRECTORY ${AKG_SOURCE_DIR}
              OUTPUT_VARIABLE EXEC_OUTPUT
              RESULT_VARIABLE RESULT)
      message("${EXEC_OUTPUT}")
      if(RESULT EQUAL 0)
        string(STRIP ${EXEC_OUTPUT} AKG_EXTEND_DIR)
        file(GLOB AKG_EXTEND ${AKG_EXTEND_DIR}/*.o)
        message("-- Find .o in ${AKG_EXTEND_DIR}")
      endif()
    endif()
  endif()

  if(EXISTS ${AKG_EXTEND_FILE})
    file(COPY ${AKG_EXTEND_FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/akg_extend)
    execute_process(COMMAND ar -x libakg_ext.a
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/akg_extend)
    file(GLOB AKG_EXTEND ${CMAKE_CURRENT_BINARY_DIR}/akg_extend/*.o)
  endif()

  if(NOT AKG_EXTEND)
    message("-- Warning: Build AKG without Ascend back-end support")
  else()
    message("-- Build AKG with Ascend back-end support")
  endif()

  if(USE_KC_AIR)
    message("-- Build with kc air")
    add_definitions(-DUSE_KC_AIR=1)
    set(TVM_RUNTIME_LINKER_LIBS kc_air)
    link_directories("${CMAKE_CURRENT_BINARY_DIR}")
  endif()

endif()
