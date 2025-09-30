include(CheckSymbolExists)
check_symbol_exists(__aarch64__ "" __CHECK_AARCH64)
check_symbol_exists(__x86_64__ "" __CHECK_X86_64)

set(AscendNpuIR_URL "https://gitee.com/ascend/ascendnpu-ir/archive/refs/tags/v0.4-beta.tar.gz")
set(AscendNpuIR_MD5 "c3f9e8fda069ce04533815f3ed6760e0")

if(__CHECK_X86_64)
  set(BishengIR_URL "https://gitee.com/ascend/ascendnpu-ir/releases/download/v0.4-beta/bishengir_x86.tar.gz")
  set(BishengIR_MD5 "ea33b239a5edd96a5285f7d5ab1bae90")
elseif(__CHECK_AARCH64)
  set(BishengIR_URL "https://gitee.com/ascend/ascendnpu-ir/releases/download/v0.4-beta/bishengir_aarch64.tar.gz")
  set(BishengIR_MD5 "9cc2569882475d1dc59da4142c0f96be")
else()
  message(FATAL_ERROR "runtime only support aarch64 and x86_64")
endif()

akg_add_pkg(bisheng_ir
        VER 0.4.0
        HEAD_ONLY ./
        URL ${BishengIR_URL}
        MD5 ${BishengIR_MD5})

akg_add_pkg(ascendnpu_ir
        VER 0.4.0
        HEAD_ONLY bishengir/include
        URL ${AscendNpuIR_URL}
        MD5 ${AscendNpuIR_MD5})

execute_process(COMMAND chmod -R +x ${bisheng_ir_ROOT}/bin
                WORKING_DIRECTORY ${bisheng_ir_ROOT}/bin
                RESULT_VARIABLE Result)
include_directories(${ascendnpu_ir_INC}
                    ${CMAKE_CURRENT_BINARY_DIR}/bisheng_ir_build/bishengir/include)
set(BISHENG_IR_INSTALL_PATH ${bisheng_ir_ROOT} CACHE STRING
    "install path for pre-build required for bishengir")
add_subdirectory(${ascendnpu_ir_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/bisheng_ir_build)
