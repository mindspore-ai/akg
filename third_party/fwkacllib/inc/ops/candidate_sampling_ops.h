/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef GE_OP_CANDIDATE_SAMPLING_OPS_H_
#define GE_OP_CANDIDATE_SAMPLING_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Generates labels for candidate sampling with \n
a learned unigram distribution.

*@par Inputs: 
*Input "true_classes" is a 2D matrix. \n
*true_classes: A "batch_size * num_true" matrix, in which each row contains \n
the IDs of the "num_true" "target_classes" in the corresponding original label.

*@par Attributes: 
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection, \n
so that all sampled candidates in a batch are unique.
*This requires some approximation to estimate the post-rejection \n
sampling probabilities.
*@li range_max: The sampler will sample integers from the interval \n
[0, range_max).
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision.

*@par Outputs: 
*@li sampled_candidates: A vector of length "num_sampled", in which each \n
element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing \n
the number of times each candidate is expected to occur in a batch of sampled \n
candidates. If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", \n
for each sampled candidate.
*representing the number of times the candidate is expected to occur \n
in a batch of sampled candidates.
* If "unique" is true, then this is a probability. \n

*@attention Constraints: \n
*ThreadUnsafeUnigramCandidateSampler runs on the Ascend AI CPU, \n
which delivers poor performance.
*/

REG_OP(ThreadUnsafeUnigramCandidateSampler)
    .INPUT(true_classes, TensorType({ DT_INT64 }))
    .OUTPUT(sampled_candidates, TensorType({ DT_INT64 }))
    .OUTPUT(true_expected_count, TensorType({ DT_FLOAT }))
    .OUTPUT(sampled_expected_count, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(num_true, Int)
    .REQUIRED_ATTR(num_sampled, Int)
    .REQUIRED_ATTR(unique, Bool)
    .REQUIRED_ATTR(range_max, Int)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(ThreadUnsafeUnigramCandidateSampler)

/**
*@brief Generates labels for candidate sampling with a learned \n
unigram distribution.

*@par Inputs: 
*true_classes: A "batch_size * num_true" matrix, in which each row contains \n
the IDs of the "num_true" "target_classes" in the corresponding original label.
*Input "true_classes" is a 2D matrix.

*@par Attributes: 
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection, \n
so that all sampled candidates in a batch are unique.
*This requires some approximation to estimate the post-rejection \n
sampling probabilities.
*@li range_max: The sampler will sample integers from the interval \n
[0, range_max).
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision.

*@par Outputs: 
*@li sampled_candidates: A vector of length "num_sampled", \n
in which each element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing the \n
number of times each candidate is expected to occur \n
in a batch of sampled candidates.
*If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", for each \n
sampled candidate representing the number of times.
* the candidate is expected to occur in a batch of sampled candidates. \n
*If "unique" is true, then this is a probability.

*@attention Constraints: \n
*UniformCandidateSampler runs on the Ascend AI CPU, \n
which delivers poor performance.
*/

REG_OP(UniformCandidateSampler)
    .INPUT(true_classes, TensorType({ DT_INT64 }))
    .OUTPUT(sampled_candidates, TensorType({ DT_INT64 }))
    .OUTPUT(true_expected_count, TensorType({ DT_FLOAT }))
    .OUTPUT(sampled_expected_count, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(num_true, Int)
    .REQUIRED_ATTR(num_sampled, Int)
    .REQUIRED_ATTR(unique, Bool)
    .REQUIRED_ATTR(range_max, Int)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(UniformCandidateSampler)

/**
*@brief Generates labels for candidate sampling with a learned \n
unigram distribution.

*@par Inputs: 
*true_classes: A "batch_size * num_true" matrix, in which each row contains \n
the IDs of the "num_true" "target_classes" in the corresponding original label.
* Input "true_classes" is a 2D matrix.

*@par Attributes: 
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection, \n
so that all sampled candidates in a batch are unique. This requires \n
some approximation to estimate the post-rejection sampling probabilities.
*@li range_max: The sampler will sample integers from the interval [0, range_max).
*@li vocab_file: Each valid line in this file (which should have a \n
CSV-like format) corresponds to a valid word ID. \n
*IDs are in sequential order, starting from num_reserved_ids.
*@li distortion: The distortion is used to skew the unigram probability \n
distribution. Each weight is first raised to the distortion's power before \n
adding to the internal unigram distribution.
*@li num_reserved_ids: Optionally some reserved IDs can be added in the range \n
[0, ..., num_reserved_ids) by the users. \n
* One use case is that a special unknown word token is used as ID 0.
*@li num_shards: A sampler can be used to sample from a subset of the \n 
original range. in order to speed up the whole computation through parallelism.
*@li shard: A sampler can be used to sample from a subset of the original \n
range in order to speed up the whole computation through parallelism.
*@li unigrams: A list of unigram counts or probabilities, one per ID in \n
sequential order.
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision.

*@par Outputs: 
*@li sampled_candidates: A vector of length "num_sampled", in which each \n
element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing the \n
number of times each candidate is expected to occur in a batch of sampled \n
candidates. If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", \n
for each sampled candidate representing the number of times the candidate is \n
expected to occur in a batch of sampled candidates. \n
If "unique" is true, then this is a probability.

*@attention Constraints: \n
* FixedUnigramCandidateSampler runs on the Ascend AI CPU, \n
which delivers poor performance.
*/

REG_OP(FixedUnigramCandidateSampler)
    .INPUT(true_classes, TensorType({ DT_INT64 }))
    .OUTPUT(sampled_candidates, TensorType({ DT_INT64 }))
    .OUTPUT(true_expected_count, TensorType({ DT_FLOAT }))
    .OUTPUT(sampled_expected_count, TensorType({ DT_FLOAT }))
    .ATTR(num_true, Int, 0)
    .ATTR(num_sampled, Int, 0)
    .ATTR(unique, Bool, false)
    .ATTR(range_max, Int, 0)
    .ATTR(vocab_file, String, "")
    .ATTR(distortion, Float, 1.0)
    .ATTR(num_reserved_ids, Int, 0)
    .ATTR(num_shards, Int, 1)
    .ATTR(shard, Int, 0)
    .REQUIRED_ATTR(unigrams, ListFloat)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(FixedUnigramCandidateSampler)

/**
*@brief Generates labels for candidate sampling with a learned \n
unigram distribution.

*@par Inputs: 
*true_classes: A "batch_size * num_true" matrix, in which each row contains \n
the IDs of the "num_true" "target_classes" in the corresponding original label.
* Input "true_classes" is a 2D matrix.

*@par Attributes: 
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection, \n
so that all sampled candidates in a batch are unique. \n
*This requires some approximation to estimate the post-rejection \n
sampling probabilities.
*@li range_max: The sampler will sample integers from the interval \n
[0, range_max).
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision.

*@par Outputs: 
*@li sampled_candidates: A vector of length "num_sampled", in which each \n
element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing \n
the number of times each candidate is expected to occur in a batch of sampled candidates. \n
*If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", for each \n
sampled candidate representing the number of times the candidate is expected \n
to occur in a batch of sampled candidates. \n
*If "unique" is true, then this is a probability.

*@attention Constraints: \n
*LearnedUnigramCandidateSampler runs on the Ascend AI CPU, which delivers \n
poor performance.
*/

REG_OP(LearnedUnigramCandidateSampler)
    .INPUT(true_classes, TensorType({ DT_INT64 }))
    .OUTPUT(sampled_candidates, TensorType({ DT_INT64 }))
    .OUTPUT(true_expected_count, TensorType({ DT_FLOAT }))
    .OUTPUT(sampled_expected_count, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(num_true, Int)
    .REQUIRED_ATTR(num_sampled, Int)
    .REQUIRED_ATTR(unique, Bool)
    .REQUIRED_ATTR(range_max, Int)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(LearnedUnigramCandidateSampler)

/**
*@brief Generates labels for candidate sampling with a log-uniform \n
distribution.

*@par Inputs: 
*true_classes: A "batch_size * num_true" matrix, in which each row contains \n
the IDs of the "num_true" "target_classes" in the corresponding original label. \n
* Input "true_classes" is a 2D matrix.

*@par Attributes: 
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection, so that all \n
sampled candidates in a batch are unique. This requires some approximation \n
to estimate the post-rejection sampling probabilities.
*@li range_max: The sampler will sample integers from the interval \n
[0, range_max).
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision.

*@par Outputs: 
*@li sampled_candidates: A vector of length "num_sampled", in which each \n
element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing \n
the number of times each candidate is expected to occur in a batch of sampled \n
candidates. If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", for each \n
sampled candidate representing the number of times the candidate is expected \n
to occur in a batch of sampled candidates. \n
*If "unique" is true, then this is a probability.

*@attention Constraints: \n
*LogUniformCandidateSampler runs on the Ascend AI CPU, which delivers \n
poor performance.
*/

REG_OP(LogUniformCandidateSampler)
    .INPUT(true_classes, TensorType({ DT_INT64 }))
    .OUTPUT(sampled_candidates, TensorType({ DT_INT64 }))
    .OUTPUT(true_expected_count, TensorType({ DT_FLOAT }))
    .OUTPUT(sampled_expected_count, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(num_true, Int)
    .REQUIRED_ATTR(num_sampled, Int)
    .REQUIRED_ATTR(unique, Bool)
    .REQUIRED_ATTR(range_max, Int)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(LogUniformCandidateSampler)

/**
*@brief Generates labels for candidate sampling with a learned \n
unigram distribution.

*@par Inputs: 
*true_classes: A "batch_size * num_true" matrix, in which each row contains \n
the IDs of the "num_true" "target_classes" in the corresponding original label. \n
* Input "true_classes" is a 2D matrix.

*@par Attributes: 
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection, \n
so that all sampled candidates in a batch are unique. This requires some \n
approximation to estimate the post-rejection sampling probabilities.
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision.

*@par Outputs: 
*@li sampled_candidates: A vector of length "num_sampled", \n
in which each element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing the \n
number of times each candidate is expected to occur in a batch of sampled candidates. \n
*If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", for each \n
sampled candidate representing the number of times the candidate is expected \n
to occur in a batch of sampled candidates. If "unique" is true, then this is a probability.

*@attention Constraints: \n
*AllCandidateSampler runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(AllCandidateSampler)
    .INPUT(true_classes, TensorType({ DT_INT64 }))
    .OUTPUT(sampled_candidates, TensorType({ DT_INT64 }))
    .OUTPUT(true_expected_count, TensorType({ DT_FLOAT }))
    .OUTPUT(sampled_expected_count, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(num_true, Int)
    .REQUIRED_ATTR(num_sampled, Int)
    .REQUIRED_ATTR(unique, Bool)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(AllCandidateSampler)

/**
*@brief Computes the "ids" of the positions in "sampled_candidates" that \n
match "true_labels".

*@par Inputs: 
* @li Input "true_classes" is a 2D matrix. \n
* @li true_classes: The "true_classes" output of UnpackSparseLabels. \n
* @li sampled_candidates: The "sampled_candidates" output of CandidateSampler. \n

*@par Attributes: 
*@li num_true: Number of true labels per context.
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision.

*@par Outputs: 
* @li indices: A vector of indices corresponding to rows of "true_candidates".
* @li ids: A vector of IDs of positions in "sampled_candidates" that match a \n
"true_label" for the row with the corresponding index in indices.
* @li weights: A vector of the same length as "indices" and "ids", in which \n
each element is -FLOAT_MAX.

*@attention Constraints: \n
*ComputeAccidentalHits runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(ComputeAccidentalHits)
    .INPUT(true_classes, TensorType({ DT_INT64 }))
    .INPUT(sampled_candidates, TensorType({ DT_INT64 }))
    .OUTPUT(indices, TensorType({ DT_INT32 }))
    .OUTPUT(ids, TensorType({ DT_INT64 }))
    .OUTPUT(weights, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(num_true, Int)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(ComputeAccidentalHits)

}  // namespace ge

#endif  // GE_OP_CANDIDATE_SAMPLING_OPS_H_
