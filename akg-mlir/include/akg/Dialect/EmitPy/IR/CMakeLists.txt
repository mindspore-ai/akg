# Add the Interfaces directory to tablegen include path so that EmitPyOps.td
# can include the interface definitions.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Interfaces)

set(LLVM_TARGET_DEFINITIONS EmitPyOps.td)
add_mlir_dialect(EmitPyOps emitpy)

set(LLVM_TARGET_DEFINITIONS EmitPyAttrs.td)
mlir_tablegen(EmitPyAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(EmitPyAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(AKGEmitPyAttrsIncGen)
add_dependencies(mlir-headers AKGEmitPyAttrsIncGen)
