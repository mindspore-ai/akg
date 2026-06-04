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

//===--------- FusionUtils.h - Utils for fusion pass ----*- C++ ---------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AKG_DIALECT_LINALG_UTILS_FUSIONUTILS_H_
#define AKG_DIALECT_LINALG_UTILS_FUSIONUTILS_H_

#include <algorithm>
#include "mlir/Dialect/Linalg/IR/Linalg.h"
namespace mlir {
namespace linalg {

static SmallVector<Value> collectFusedYieldValues(linalg::YieldOp producerYieldOp, linalg::YieldOp consumerYieldOp,
                                                  IRMapping &mapper,
                                                  llvm::SmallDenseSet<int> &preservedProducerResults) {
  SmallVector<Value> fusedYieldValues;
  fusedYieldValues.reserve(producerYieldOp.getNumOperands() + consumerYieldOp.getNumOperands());
  for (const auto &producerYieldVal : llvm::enumerate(producerYieldOp.getOperands())) {
    if (preservedProducerResults.count(producerYieldVal.index()) != static_cast<size_t>(0)) {
      fusedYieldValues.push_back(mapper.lookupOrDefault(producerYieldVal.value()));
    }
  }
  for (auto consumerYieldVal : consumerYieldOp.getOperands()) {
    fusedYieldValues.push_back(mapper.lookupOrDefault(consumerYieldVal));
  }
  return fusedYieldValues;
}

static void spliceFusedBlockArgs(Block *fusedBlock, Block &producerBlock, Block &consumerBlock, OpOperand *fusedOperand,
                                 IRMapping &mapper, LinalgOp producer, LinalgOp consumer,
                                 llvm::SmallDenseSet<int> &preservedProducerResults) {
  for (BlockArgument bbArg : consumerBlock.getArguments().take_front(fusedOperand->getOperandNumber())) {
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }
  for (BlockArgument bbArg : producerBlock.getArguments().take_front(producer.getNumDpsInputs())) {
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }
  for (BlockArgument bbArg : consumerBlock.getArguments()
                               .take_front(consumer.getNumDpsInputs())
                               .drop_front(fusedOperand->getOperandNumber() + 1)) {
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }
  for (const auto &bbArg : llvm::enumerate(producerBlock.getArguments().take_back(producer.getNumDpsInits()))) {
    if (preservedProducerResults.count(bbArg.index()) == static_cast<size_t>(0)) continue;
    mapper.map(bbArg.value(), fusedBlock->addArgument(bbArg.value().getType(), bbArg.value().getLoc()));
  }
  for (BlockArgument bbArg : consumerBlock.getArguments().take_back(consumer.getNumDpsInits())) {
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }
}

/// Generate the region of the fused tensor operation. The region of the fused
/// op must be empty.
template <typename PRODUCEROP, typename CONSUMEROP>
void generateFusedElementwiseOpRegion(RewriterBase &rewriter, CONSUMEROP fusedOp, AffineMap consumerToProducerLoopsMap,
                                      OpOperand *fusedOperand, llvm::SmallDenseSet<int> &preservedProducerResults) {
  auto producer = cast<PRODUCEROP>(fusedOperand->get().getDefiningOp());
  auto consumer = cast<CONSUMEROP>(fusedOperand->getOwner());
  Block &producerBlock = producer->getRegion(0).front();
  Block &consumerBlock = consumer->getRegion(0).front();
  Block *fusedBlock = new Block();
  fusedOp.getRegion().push_back(fusedBlock);
  IRMapping mapper;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(fusedBlock);

  if (producer.hasIndexSemantics()) {
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
  assert(consumer.isDpsInput(fusedOperand) && "expected producer of input operand");
  spliceFusedBlockArgs(fusedBlock, producerBlock, consumerBlock, fusedOperand, mapper, producer, consumer,
                       preservedProducerResults);

  for (auto &op : producerBlock.without_terminator()) {
    if (!isa<IndexOp>(op)) {
      (void)rewriter.clone(op, mapper);
    }
  }
  auto producerYieldOp = cast<linalg::YieldOp>(producerBlock.getTerminator());
  unsigned producerResultNumber = cast<OpResult>(fusedOperand->get()).getResultNumber();
  Value replacement = mapper.lookupOrDefault(producerYieldOp.getOperand(producerResultNumber));

  if (replacement == producerYieldOp.getOperand(producerResultNumber)) {
    if (auto bb = dyn_cast<BlockArgument>(replacement)) {
      assert(bb.getOwner() != &producerBlock && "yielded block argument must have been mapped");
    } else {
      assert(!producer->isAncestor(replacement.getDefiningOp()) && "yielded value must have been mapped");
    }
  }
  mapper.map(consumerBlock.getArgument(fusedOperand->getOperandNumber()), replacement);
  for (auto &op : consumerBlock.without_terminator()) {
    (void)rewriter.clone(op, mapper);
  }

  auto consumerYieldOp = cast<linalg::YieldOp>(consumerBlock.getTerminator());
  rewriter.create<YieldOp>(fusedOp.getLoc(),
                           collectFusedYieldValues(producerYieldOp, consumerYieldOp, mapper, preservedProducerResults));
  assert(fusedBlock->getNumArguments() == fusedOp.getNumOperands() && "Ill-formed GenericOp region");
}

}  // namespace linalg
}  // namespace mlir

#endif  // AKG_DIALECT_LINALG_UTILS_FUSIONUTILS_H_
