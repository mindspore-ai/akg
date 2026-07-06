/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <unordered_map>
#include <utility>

#include "akg/Target/PTX/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

using namespace llvm;  // NOLINT(build/namespaces)
using namespace mlir;  // NOLINT(build/namespaces)

namespace {

static constexpr unsigned kOptLevelDefault = 2;
static constexpr unsigned kOptLevelAggressive = 3;
static constexpr int kMaxSizeLevel = 2;
static llvm::CodeGenOptLevel LLVMCodeGenOpt(unsigned optLevel) {
  static const std::unordered_map<unsigned, llvm::CodeGenOptLevel> kOptLevelMap = {
    {0, llvm::CodeGenOptLevel::None},
    {1, llvm::CodeGenOptLevel::Less},
    {kOptLevelDefault, llvm::CodeGenOptLevel::Default},
    {kOptLevelAggressive, llvm::CodeGenOptLevel::Aggressive}};
  auto it = kOptLevelMap.find(optLevel);
  return it != kOptLevelMap.end() ? it->second : llvm::CodeGenOptLevel::Aggressive;
}

static llvm::OptimizationLevel mapToLevel(unsigned optLevel) {
  static const std::unordered_map<unsigned, llvm::OptimizationLevel> kOptLevelMap = {
    {0, llvm::OptimizationLevel::O0},
    {1, llvm::OptimizationLevel::O1},
    {kOptLevelDefault, llvm::OptimizationLevel::O2},
    {kOptLevelAggressive, llvm::OptimizationLevel::O3}};
  auto it = kOptLevelMap.find(optLevel);
  if (it == kOptLevelMap.end()) {
    llvm_unreachable("Invalid optimization level!");
  }
  return it->second;
}

class GPUKernelToPTX : public PassWrapper<GPUKernelToPTX, OperationPass<gpu::GPUModuleOp>> {
 public:
  // cppcheck-suppress unknownMacro
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUKernelToPTX)

  GPUKernelToPTX(unsigned opt, std::string libdeviceFile, std::string triple, std::string chip, std::string features,
                 std::string &targetISA)
      : optLevelAsInt(opt),
        libdeviceFile(std::move(libdeviceFile)),
        triple(std::move(triple)),
        chip(std::move(chip)),
        features(std::move(features)),
        targetISA(targetISA) {}

  void runOnOperation() override;

 private:
  void translateToISA(llvm::Module &llvmModule, llvm::TargetMachine &targetMachine);
  std::unique_ptr<llvm::TargetMachine> createTargetMachine();
  LogicalResult linkLibdevice(llvm::Module &llvmModule, llvm::LLVMContext &llvmContext);

  unsigned optLevelAsInt;
  std::string libdeviceFile;
  std::string triple;
  std::string chip;
  std::string features;
  std::string &targetISA;
};

void GPUKernelToPTX::runOnOperation() {
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(getOperation(), llvmContext, "LLVMDialectModule");

  if (!llvmModule) {
    return signalPassFailure();
  }

  if (failed(linkLibdevice(*llvmModule, llvmContext))) {
    return signalPassFailure();
  }

  std::unique_ptr<llvm::TargetMachine> targetMachine = createTargetMachine();
  if (!targetMachine) {
    return signalPassFailure();
  }

  translateToISA(*llvmModule, *targetMachine);
}

void GPUKernelToPTX::translateToISA(llvm::Module &llvmModule, llvm::TargetMachine &targetMachine) {
  llvmModule.setDataLayout(targetMachine.createDataLayout());

  llvm::raw_string_ostream stream(targetISA);
  llvm::buffer_ostream pstream(stream);

  llvm::OptimizationLevel optLevel = mapToLevel(optLevelAsInt);
  llvm::PassBuilder pB(&targetMachine);

  llvm::LoopAnalysisManager lAM;
  llvm::FunctionAnalysisManager fAM;
  llvm::CGSCCAnalysisManager cGAM;
  llvm::ModuleAnalysisManager mAM;

  pB.registerModuleAnalyses(mAM);
  pB.registerCGSCCAnalyses(cGAM);
  pB.registerFunctionAnalyses(fAM);
  pB.registerLoopAnalyses(lAM);
  pB.crossRegisterProxies(lAM, fAM, cGAM, mAM);

  llvm::FunctionPassManager fPM = pB.buildFunctionSimplificationPipeline(optLevel, llvm::ThinOrFullLTOPhase::None);
  llvm::ModulePassManager mPM = pB.buildPerModuleDefaultPipeline(optLevel);

  (void)fAM.registerPass([&targetMachine] { return targetMachine.getTargetIRAnalysis(); });

  fPM.addPass(llvm::VerifierPass());  // Verify that input is correct
  if (optLevel.getSpeedupLevel() > 1 && optLevel.getSizeLevel() < kMaxSizeLevel) {
    fPM.addPass(llvm::LoopVectorizePass());
    fPM.addPass(llvm::SLPVectorizerPass());
  }
  fPM.addPass(llvm::DCEPass());
  mPM.addPass(llvm::AlwaysInlinerPass());

  (void)mPM.run(llvmModule, mAM);
  for (auto &f : llvmModule) {
    if (!f.empty()) {
      (void)fPM.run(f, fAM);
    }
  }

  llvm::legacy::PassManager codegenPasses;
  codegenPasses.add(llvm::createVerifierPass());
  (void)targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr, llvm::CodeGenFileType::AssemblyFile);
  (void)codegenPasses.run(llvmModule);
}

std::unique_ptr<llvm::TargetMachine> GPUKernelToPTX::createTargetMachine() {
  const Location loc = getOperation().getLoc();
  std::string error;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);

  if (target == nullptr) {
    (void)emitError(loc, Twine("failed to lookup target: ") + error);
    return {};
  }

  llvm::TargetMachine *machine =
    target->createTargetMachine(triple, chip, features, {}, {}, std::nullopt, LLVMCodeGenOpt(optLevelAsInt));
  if (machine == nullptr) {
    (void)emitError(loc, "failed to create target machine");
    return {};
  }

  return std::unique_ptr<llvm::TargetMachine>{machine};
}

LogicalResult GPUKernelToPTX::linkLibdevice(llvm::Module &llvmModule, llvm::LLVMContext &llvmContext) {
  if (libdeviceFile.empty()) {
    llvm::errs() << "Fatal: unable to locate libdevice.10.bc\n";
    return failure();
  }
  std::string errorMessage;
  auto libdeviceBuf = openInputFile(libdeviceFile, &errorMessage);
  if (!libdeviceBuf) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto moduleOrErr = llvm::getOwningLazyBitcodeModule(std::move(libdeviceBuf), llvmContext);
  if (!moduleOrErr) {
    llvm::errs() << "Failed to load libdevice bitcode from " << libdeviceFile << "\n";
    return failure();
  }

  std::unique_ptr<llvm::Module> libdeviceModule = std::move(moduleOrErr.get());
  for (llvm::Function &F : *libdeviceModule.get()) {
    if (F.isIntrinsic()) {
      continue;
    }

    llvm::AttrBuilder FuncAttrs(llvmContext);
    // FramePointerKind = "all"
    (void)FuncAttrs.addAttribute("frame-pointer", "all");
    (void)FuncAttrs.addAttribute("less-precise-fpmad", "false");
    (void)FuncAttrs.addAttribute("no-trapping-math", "true");
    (void)FuncAttrs.addAttribute(llvm::Attribute::Convergent);
    // no exceptions for cuda device code
    (void)FuncAttrs.addAttribute(llvm::Attribute::NoUnwind);

    F.addFnAttrs(FuncAttrs);
  }

  // libdevice module is of an ``internalize'' module
  // LinkFlags = LinkOnlyNeeded
  if (llvm::Linker::linkModules(llvmModule, std::move(libdeviceModule),
                                static_cast<unsigned>(llvm::Linker::Flags::LinkOnlyNeeded),
                                [](llvm::Module &M, const llvm::StringSet<> &GS) {
                                  (void)llvm::internalizeModule(M, [&GS](const llvm::GlobalValue &GV) {
                                    return !GV.hasName() || (GS.count(GV.getName()) == 0);
                                  });
                                })) {
    llvm::errs() << "failed to link libdevice module\n";
    return failure();
  }
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> mlir::createSerializeToPTXPass(
  unsigned optLevel, const std::string &libdeviceFile, const std::string &triple, const std::string &chip,
  const std::string &features, std::string &targetISA) {
  return std::make_unique<GPUKernelToPTX>(optLevel, libdeviceFile, triple, chip, features, targetISA);
}
