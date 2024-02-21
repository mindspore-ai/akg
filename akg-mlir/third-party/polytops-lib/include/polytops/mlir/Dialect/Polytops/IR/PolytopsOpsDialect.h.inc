/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace polytops {
namespace mlir {

class PolytopsDialect : public ::mlir::Dialect {
  explicit PolytopsDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~PolytopsDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("polytops");
  }
};
} // namespace mlir
} // namespace polytops
MLIR_DECLARE_EXPLICIT_TYPE_ID(::polytops::mlir::PolytopsDialect)
