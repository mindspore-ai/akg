add_mlir_dialect(Mfuse mfuse)
add_mlir_doc(Mfuse Mfuse Dialect/ -gen-op-doc)
add_mlir_interface(MfuseInterfaces)
add_dependencies(MLIRMfuseIncGen MLIRMfuseInterfacesIncGen)

# Additional TableGen targets for attributes and types
set(LLVM_TARGET_DEFINITIONS Mfuse.td)
mlir_tablegen(MfuseAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=mfuse)
mlir_tablegen(MfuseAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=mfuse)
mlir_tablegen(MfuseTypes.h.inc -gen-typedef-decls -typedefs-dialect=mfuse)
mlir_tablegen(MfuseTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=mfuse)
mlir_tablegen(MfuseEnums.h.inc -gen-enum-decls)
mlir_tablegen(MfuseEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMfuseExtraIncGen)

# Include Transforms subdirectory for pass generation
add_subdirectory(Transforms)
