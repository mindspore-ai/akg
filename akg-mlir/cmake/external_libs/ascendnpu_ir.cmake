set(REQ_URL "https://gitcode.com/Ascend/AscendNPU-IR.git")
set(VERSION "v1.1.0-post2")
set(SHA256 "e4633e70f812b7c483768fdcc850c6077a3727e1") #not used

set(ascendnpu_ir_options
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_BUILD_TYPE=Release
    -DBISHENGIR_BUILD_STANDALONE_IR_ONLY=ON
    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
    -DLLVM_DIR=${LLVM_DIR}
    -DMLIR_DIR=${MLIR_DIR}
)

set(BiShengIRLibs
    BiShengIRAnnotationDialect
    BiShengIRAnnotationTransforms
    BiShengIRDialectUtils
    BiShengIRHACCDialect
    BiShengIRHFusionDialect
    BiShengIRHIVMDialect
    BiShengIRHIVMTransforms
    BiShengIRHIVMUtils
    BiShengIRHACCUtils
    BiShengIRMathExtDialect
    BiShengIRMemRefDialect
    BiShengIRSymbolDialect
    BiShengIRTensorDialect
    BiShengIRMemRefExtDialect
    BiShengIRHACCTransforms
    BiShengIRSCFTransforms
    BiShengIRScopeDialect
    BiShengIRSCFUtils
    BiShengIRScopeTransforms
    BiShengIRArithToAffine
    BiShengIRHIVMToStandard)

akg_add_pkg(ascendnpu-ir
    VER ${VERSION}
    LIBS ${BiShengIRLibs}
    GIT_REPOSITORY ${REQ_URL}
    GIT_TAG ${VERSION}
    SHA256 ${SHA256}
    PATCHES ${CMAKE_SOURCE_DIR}/third-party/patch/ascendnpu-ir/0001-adapter-akg.patch
    CMAKE_OPTION ${ascendnpu_ir_options}
    CUSTOM_CMAKE_GENERATOR Ninja)

set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)
include_directories(${ascendnpu-ir_INC})

foreach(libs IN LISTS BiShengIRLibs)
    add_library(${libs} ALIAS ascendnpu-ir::${libs})
endforeach()
