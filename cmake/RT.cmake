include(CheckSymbolExists)
check_symbol_exists(__aarch64__ "" __CHECK_AARCH64)
check_symbol_exists(__x86_64__ "" __CHECK_X86_64)
set(AUTODIFF_SITE "http://autodiff.huawei.com:8080")
if(USE_CCE_RT)
  message("-- Build with cce runtime")
  
  if(NOT __CHECK_AARCH64 AND NOT __CHECK_X86_64)
    message(FATAL_ERROR "runtime only support aarch64 and x86_64")
  endif()

  add_definitions(-DUSE_CCE_RT=1)
  set(TVM_RUNTIME_LINKER_LIBS libruntime.so)
  link_directories(/usr/local/Ascend/fwkacllib/lib64)

elseif(USE_CCE_RT_SIM)
  message("-- Build with cce runtime(camodel), Only Support AMD64")
  
  if(NOT __CHECK_X86_64)
    message(FATAL_ERROR "camodel only support x86_64")
  endif()
  
  add_definitions(-DUSE_CCE_RT_SIM=1)
  set(TVM_RUNTIME_LINKER_LIBS libcamodel.so libslog.so libc_sec.so
                              libruntime_camodel.so libtsch_camodel.so)
  
  foreach(LIB IN LISTS TVM_RUNTIME_LINKER_LIBS)
    file(DOWNLOAD ${AUTODIFF_SITE}/x86_64/camodel/${LIB}
        ${CMAKE_CURRENT_BINARY_DIR}/x86_64/camodel/${LIB})
  endforeach()

   link_directories(
       "${CMAKE_CURRENT_BINARY_DIR}/x86_64/camodel")

elseif(USE_KC_AIR)
  message("-- Build with kc air")
  if(NOT __CHECK_AARCH64 AND NOT __CHECK_X86_64)
    message(FATAL_ERROR "-- now kc air only support amd64 and x86_64")
  endif()
  if(__CHECK_AARCH64)
      file(DOWNLOAD ${AUTODIFF_SITE}/aarch64/kc_air/libkc_air.so
          ${CMAKE_CURRENT_BINARY_DIR}/aarch64/libkc_air.so)
      link_directories("${CMAKE_CURRENT_BINARY_DIR}/aarch64")
  endif()
  if(__CHECK_X86_64)
      file(DOWNLOAD ${AUTODIFF_SITE}/x86_64/kc_air/libkc_air.so
          ${CMAKE_CURRENT_BINARY_DIR}/x86_64/libkc_air.so)
      link_directories("${CMAKE_CURRENT_BINARY_DIR}/x86_64")
  endif()
  add_definitions(-DUSE_CCE_RT=1)
  add_definitions(-DUSE_KC_AIR=1)
  set(TVM_RUNTIME_LINKER_LIBS kc_air)
else()
  message("-- Build without runtime support")
  add_definitions(-DUSE_CCE_RT_STUB=1)
  add_definitions(-DUSE_CCE_RT=1)
endif()
