set(AscendNPU-IR_URL "https://gitcode.com/Ascend/AscendNPU-IR.git")
set(AscendNPU-IR_TAG "e4633e70f812b7c483768fdcc850c6077a3727e1")
set(AscendNPU-IR_MD5 "e4633e70f812b7c483768fdcc850c6077a3727e1") #not used

set(ascendnpu_ir_options
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_BUILD_TYPE=Release
    -DBISHENGIR_BUILD_STANDALONE_IR_ONLY=ON
    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
)

akg_add_pkg(ascendnpu-ir
            VER ${AscendNPU-IR_TAG}
            LIBS
              BiShengIRAnnotationDialect
              BiShengIRDialectUtils
              BiShengIRHACCDialect
              BiShengIRHFusionDialect
              BiShengIRHIVMDialect
              BiShengIRHIVMUtils
              BiShengIRMathExtDialect
              BiShengIRMemRefDialect
              BiShengIRSymbolDialect
              BiShengIRTensorDialect
              BiShengIRMemRefExtDialect
            GIT_REPOSITORY ${AscendNPU-IR_URL}
            GIT_TAG ${AscendNPU-IR_TAG}
            MD5 ${AscendNPU-IR_MD5}
            PATCHES ${CMAKE_SOURCE_DIR}/third-party/patch/ascendnpu-ir/0001-adapter-akg.patch
            CMAKE_OPTION ${ascendnpu_ir_options})

set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)
include_directories(${ascendnpu-ir_INC})

add_library(BiShengIRAnnotationDialect ALIAS ascendnpu-ir::BiShengIRAnnotationDialect)
add_library(BiShengIRDialectUtils ALIAS ascendnpu-ir::BiShengIRDialectUtils)
add_library(BiShengIRHACCDialect ALIAS ascendnpu-ir::BiShengIRHACCDialect)
add_library(BiShengIRHFusionDialect ALIAS ascendnpu-ir::BiShengIRHFusionDialect)
add_library(BiShengIRHIVMDialect ALIAS ascendnpu-ir::BiShengIRHIVMDialect)
add_library(BiShengIRHIVMUtils ALIAS ascendnpu-ir::BiShengIRHIVMUtils)
add_library(BiShengIRMathExtDialect ALIAS ascendnpu-ir::BiShengIRMathExtDialect)
add_library(BiShengIRMemRefDialect ALIAS ascendnpu-ir::BiShengIRMemRefDialect)
add_library(BiShengIRSymbolDialect ALIAS ascendnpu-ir::BiShengIRSymbolDialect)
add_library(BiShengIRTensorDialect ALIAS ascendnpu-ir::BiShengIRTensorDialect)
add_library(BiShengIRMemRefExtDialect ALIAS ascendnpu-ir::BiShengIRMemRefExtDialect)
