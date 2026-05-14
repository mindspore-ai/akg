add_mlir_dialect(Dvm dvm)
add_mlir_doc(Dvm Dvm Dialect/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Dvm.td)
mlir_tablegen(DvmEnums.h.inc -gen-enum-decls)
mlir_tablegen(DvmEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRDvmExtraIncGen)
