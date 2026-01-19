add_mlir_dialect(Muse muse)
add_mlir_doc(Muse Muse Dialect/ -gen-op-doc)

# Additional TableGen targets for attributes and types
set(LLVM_TARGET_DEFINITIONS Muse.td)
mlir_tablegen(MuseAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=muse)
mlir_tablegen(MuseAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=muse)
mlir_tablegen(MuseTypes.h.inc -gen-typedef-decls -typedefs-dialect=muse)
mlir_tablegen(MuseTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=muse)
mlir_tablegen(MuseEnums.h.inc -gen-enum-decls)
mlir_tablegen(MuseEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMuseExtraIncGen)

# Include Transforms subdirectory for pass generation
add_subdirectory(Transforms)
