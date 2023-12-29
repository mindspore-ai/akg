include(CheckSymbolExists)
check_symbol_exists(__aarch64__ "" __CHECK_AARCH64)
check_symbol_exists(__x86_64__ "" __CHECK_X86_64)

add_definitions(-DUSE_CCE_RT=1)

if(ENABLE_D)
  if(NOT __CHECK_AARCH64 AND NOT __CHECK_X86_64)
    message(FATAL_ERROR "runtime only support aarch64 and x86_64")
  endif()

  if(USE_KC_AIR)
    message("-- Build with kc air")
    add_definitions(-DUSE_KC_AIR=1)
    set(TVM_RUNTIME_LINKER_LIBS kc_air)
    link_directories("${CMAKE_CURRENT_BINARY_DIR}")
  endif()
endif()
