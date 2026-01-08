# Generate pass headers
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MuseTransforms)
add_public_tablegen_target(MLIRMuseTransformsPassIncGen)
