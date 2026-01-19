# Generate pass headers
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MFusionConversion)
add_public_tablegen_target(MFusionConversionPassIncGen)
