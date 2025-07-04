//===--------- FusionUtils.h - Utils for fusion pass ----*- C++ ---------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMPILER_INCLUDE_AKG_DIALECT_LINALG_UTILS_FUSIONUTILS_H_
#define COMPILER_INCLUDE_AKG_DIALECT_LINALG_UTILS_FUSIONUTILS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
namespace mlir {
namespace linalg {

/// Generate the region of the fused tensor operation. The region of the fused
/// op must be empty.
template <typename PRODUCEROP, typename CONSUMEROP>
void generateFusedElementwiseOpRegion(RewriterBase &rewriter, CONSUMEROP fusedOp, AffineMap consumerToProducerLoopsMap,
                                      OpOperand *fusedOperand, llvm::SmallDenseSet<int> &preservedProducerResults) {
  auto producer = cast<PRODUCEROP>(fusedOperand->get().getDefiningOp());
  auto consumer = cast<CONSUMEROP>(fusedOperand->getOwner());
  // Build the region of the fused op.
  Block &producerBlock = producer->getRegion(0).front();
  Block &consumerBlock = consumer->getRegion(0).front();
  Block *fusedBlock = new Block();
  fusedOp.getRegion().push_back(fusedBlock);
  IRMapping mapper;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(fusedBlock);

  // 2. Add an index operation for every fused loop dimension and use the
  // `consumerToProducerLoopsMap` to map the producer indices.
  if (producer.hasIndexSemantics()) {
    // Add an index operation for every fused loop dimension.
    unsigned numFusedOpLoops = std::max(producer.getNumLoops(), consumer.getNumLoops());
    SmallVector<Value> fusedIndices;
    fusedIndices.reserve(numFusedOpLoops);
    llvm::transform(llvm::seq<uint64_t>(0, numFusedOpLoops), std::back_inserter(fusedIndices),
                    [&](uint64_t dim) { return rewriter.create<IndexOp>(producer.getLoc(), dim); });
    for (IndexOp indexOp : llvm::make_early_inc_range(producerBlock.getOps<IndexOp>())) {
      Value newIndex = rewriter.create<mlir::affine::AffineApplyOp>(
        producer.getLoc(), consumerToProducerLoopsMap.getSubMap(indexOp.getDim()), fusedIndices);
      mapper.map(indexOp.getResult(), newIndex);
    }
  }
  // TODO: allow fusing the producer of an output operand.
  assert(consumer.isDpsInput(fusedOperand) && "expected producer of input operand");
  // 3. Consumer input operands up to consumerIdx (exclusive).
  for (BlockArgument bbArg : consumerBlock.getArguments().take_front(fusedOperand->getOperandNumber())) {
    // input assumption.
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }

  // Replacing consumerIdx requires getting the cloned, yielded, value from
  // the (cloned) producer block. This happens in step 9.

  // 4. Splice in producer's input operands.
  for (BlockArgument bbArg : producerBlock.getArguments().take_front(producer.getNumDpsInputs())) {
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }

  // 5. Remaining consumer's input operands (drop past index `consumerIdx`).
  for (BlockArgument bbArg : consumerBlock.getArguments()
                               .take_front(consumer.getNumDpsInputs())
                               .drop_front(fusedOperand->getOperandNumber() + 1)) {
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }

  // 6. All of the producer's output operands
  for (const auto &bbArg : llvm::enumerate(producerBlock.getArguments().take_back(producer.getNumDpsInits()))) {
    if (preservedProducerResults.count(bbArg.index()) == (size_t)0) {
      continue;
    }
    mapper.map(bbArg.value(), fusedBlock->addArgument(bbArg.value().getType(), bbArg.value().getLoc()));
  }

  // 7. All of consumer's output operands.
  for (BlockArgument bbArg : consumerBlock.getArguments().take_back(consumer.getNumDpsInits())) {
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }

  // 8. Clone all producer operations except for the yield and index operations
  // to the fused operation.
  for (auto &op : producerBlock.without_terminator()) {
    if (!isa<IndexOp>(op)) {
      (void)rewriter.clone(op, mapper);
    }
  }
  // 9. Now we can map the consumerBlock's `consumerIdx` block argument. Just
  // forward the yield operand.
  auto producerYieldOp = cast<linalg::YieldOp>(producerBlock.getTerminator());
  unsigned producerResultNumber = fusedOperand->get().cast<OpResult>().getResultNumber();
  Value replacement = mapper.lookupOrDefault(producerYieldOp.getOperand(producerResultNumber));

  // Sanity checks, if replacement is not already in the mapper then it must be
  // produced outside.
  if (replacement == producerYieldOp.getOperand(producerResultNumber)) {
    if (auto bb = replacement.dyn_cast<BlockArgument>()) {
      assert(bb.getOwner() != &producerBlock && "yielded block argument must have been mapped");
    } else {
      assert(!producer->isAncestor(replacement.getDefiningOp()) && "yielded value must have been mapped");
    }
  }
  mapper.map(consumerBlock.getArgument(fusedOperand->getOperandNumber()), replacement);
  // 10. Clone operations from the consumer to the fused op.
  for (auto &op : consumerBlock.without_terminator()) {
    (void)rewriter.clone(op, mapper);
  }

  // 11. Include the final yield (which is the remapped values for all the
  // yield)
  auto consumerYieldOp = cast<linalg::YieldOp>(consumerBlock.getTerminator());
  SmallVector<Value> fusedYieldValues;
  fusedYieldValues.reserve(producerYieldOp.getNumOperands() + consumerYieldOp.getNumOperands());
  for (const auto &producerYieldVal : llvm::enumerate(producerYieldOp.getOperands())) {
    if (preservedProducerResults.count(producerYieldVal.index()) != (size_t)0) {
      fusedYieldValues.push_back(mapper.lookupOrDefault(producerYieldVal.value()));
    }
  }
  for (auto consumerYieldVal : consumerYieldOp.getOperands()) {
    fusedYieldValues.push_back(mapper.lookupOrDefault(consumerYieldVal));
  }
  rewriter.create<YieldOp>(fusedOp.getLoc(), fusedYieldValues);

  // Sanity checks.
  assert(fusedBlock->getNumArguments() == fusedOp.getNumOperands() && "Ill-formed GenericOp region");
}

}  // namespace linalg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_LINALG_UTILS_FUSIONUTILS_H_
