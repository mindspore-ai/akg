add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AKGNPUVector)
add_public_tablegen_target(AKGNPUVectorPassIncGen)