set(AscendNPU-IR_URL "https://gitcode.com/Ascend/AscendNPU-IR.git")
set(AscendNPU-IR_TAG "e4633e70f812b7c483768fdcc850c6077a3727e1")
set(AscendNPU-IR_MD5 "e4633e70f812b7c483768fdcc850c6077a3727e1") #not used

akg_add_pkg(ascendnpu-ir
            HEAD_ONLY bishengir/include
            GIT_REPOSITORY ${AscendNPU-IR_URL}
            GIT_TAG ${AscendNPU-IR_TAG}
            MD5 ${AscendNPU-IR_MD5}
            PATCHES ${CMAKE_SOURCE_DIR}/third-party/patch/ascendnpu-ir/0001-adapter-akg.patch)

set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)
include_directories(${ascendnpu-ir_INC})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/ascendnpu-ir/bishengir/include)
add_subdirectory(${ascendnpu-ir_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/ascendnpu-ir)
