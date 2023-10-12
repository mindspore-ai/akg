/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

/*!
 * \file candidate_sampling_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CANDIDATE_SAMPLING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CANDIDATE_SAMPLING_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Generates labels for candidate sampling with
a learned unigram distribution. \n

*@par Inputs:
*Input "true_classes" is a 2D matrix.
*true_classes: A "batch_size * num_true" matrix, in which each row contains
the IDs of the "num_true" "target_classes" in the corresponding original label. \n

*@par Attributes:
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection,
so that all sampled candidates in a batch are unique.
*This requires some approximation to estimate the post-rejection
sampling probabilities.
*@li range_max: The sampler will sample integers from the interval
[0, range_max).
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision. \n

*@par Outputs:
*@li sampled_candidates: A vector of length "num_sampled", in which each
element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing
the number of times each candidate is expected to occur in a batch of sampled
candidates. If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled",
for each sampled candidate.
*representing the number of times the candidate is expected to occur
in a batch of sampled candidates.
* If "unique" is true, then this is a probability.

*@attention Constraints:
*ThreadUnsafeUnigramCandidateSampler runs on the Ascend AI CPU,
which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ThreadUnsafeUnigramCandidateSampler. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
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
*@brief Generates labels for candidate sampling with a learned
unigram distribution. \n

*@par Inputs:
*true_classes: A "batch_size * num_true" matrix, in which each row contains
the IDs of the "num_true" "target_classes" in the corresponding original label.
*Input "true_classes" is a 2D matrix. \n

*@par Attributes:
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection,
so that all sampled candidates in a batch are unique.
*This requires some approximation to estimate the post-rejection
sampling probabilities.
*@li range_max: The sampler will sample integers from the interval
[0, range_max).
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision. \n

*@par Outputs:
*@li sampled_candidates: A vector of length "num_sampled",
in which each element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing the
number of times each candidate is expected to occur
in a batch of sampled candidates.
*If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", for each
sampled candidate representing the number of times.
* the candidate is expected to occur in a batch of sampled candidates.
*If "unique" is true, then this is a probability. \n

*@attention Constraints:
*UniformCandidateSampler runs on the Ascend AI CPU,
which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator UniformCandidateSampler. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
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
*@brief Generates labels for candidate sampling with a learned
unigram distribution. \n

*@par Inputs:
*true_classes: A "batch_size * num_true" matrix, in which each row contains
the IDs of the "num_true" "target_classes" in the corresponding original label.
* Input "true_classes" is a 2D matrix. \n

*@par Attributes:
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection,
so that all sampled candidates in a batch are unique. This requires
some approximation to estimate the post-rejection sampling probabilities.
*@li range_max: The sampler will sample integers from the interval [0, range_max).
*@li vocab_file: Each valid line in this file (which should have a
CSV-like format) corresponds to a valid word ID.
*IDs are in sequential order, starting from num_reserved_ids.
*@li distortion: The distortion is used to skew the unigram probability
distribution. Each weight is first raised to the distortion's power before
adding to the internal unigram distribution.
*@li num_reserved_ids: Optionally some reserved IDs can be added in the range
[0, ..., num_reserved_ids) by the users.
* One use case is that a special unknown word token is used as ID 0.
*@li num_shards: A sampler can be used to sample from a subset of the
original range. in order to speed up the whole computation through parallelism.
*@li shard: A sampler can be used to sample from a subset of the original
range in order to speed up the whole computation through parallelism.
*@li unigrams: A list of unigram counts or probabilities, one per ID in
sequential order.
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision. \n

*@par Outputs:
*@li sampled_candidates: A vector of length "num_sampled", in which each
element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing the
number of times each candidate is expected to occur in a batch of sampled
candidates. If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled",
for each sampled candidate representing the number of times the candidate is
expected to occur in a batch of sampled candidates.
If "unique" is true, then this is a probability. \n

*@attention Constraints:
* FixedUnigramCandidateSampler runs on the Ascend AI CPU,
which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator FixedUnigramCandidateSampler. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
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
*@brief Generates labels for candidate sampling with a learned
unigram distribution. \n

*@par Inputs:
*true_classes: A "batch_size * num_true" matrix, in which each row contains
the IDs of the "num_true" "target_classes" in the corresponding original label.
* Input "true_classes" is a 2D matrix. \n

*@par Attributes:
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection,
so that all sampled candidates in a batch are unique.
*This requires some approximation to estimate the post-rejection
sampling probabilities.
*@li range_max: The sampler will sample integers from the interval
[0, range_max).
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision. \n

*@par Outputs:
*@li sampled_candidates: A vector of length "num_sampled", in which each
element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing
the number of times each candidate is expected to occur in a batch of sampled candidates.
*If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", for each
sampled candidate representing the number of times the candidate is expected
to occur in a batch of sampled candidates.
*If "unique" is true, then this is a probability. \n

*@attention Constraints:
*LearnedUnigramCandidateSampler runs on the Ascend AI CPU, which delivers
poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LearnedUnigramCandidateSampler. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
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
*@brief Generates labels for candidate sampling with a log-uniform
distribution. \n

*@par Inputs:
*true_classes: A "batch_size * num_true" matrix, in which each row contains
the IDs of the "num_true" "target_classes" in the corresponding original label.
* Input "true_classes" is a 2D matrix. \n

*@par Attributes:
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection, so that all
sampled candidates in a batch are unique. This requires some approximation
to estimate the post-rejection sampling probabilities.
*@li range_max: The sampler will sample integers from the interval
[0, range_max).
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision. \n

*@par Outputs:
*@li sampled_candidates: A vector of length "num_sampled", in which each
element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing
the number of times each candidate is expected to occur in a batch of sampled
candidates. If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", for each
sampled candidate representing the number of times the candidate is expected
to occur in a batch of sampled candidates.
*If "unique" is true, then this is a probability. \n

*@attention Constraints:
*LogUniformCandidateSampler runs on the Ascend AI CPU, which delivers
poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogUniformCandidateSampler. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
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
*@brief Generates labels for candidate sampling with a learned
unigram distribution. \n

*@par Inputs:
*true_classes: A "batch_size * num_true" matrix, in which each row contains
the IDs of the "num_true" "target_classes" in the corresponding original label.
* Input "true_classes" is a 2D matrix. \n

*@par Attributes:
*@li num_true: Number of true labels per context.
*@li num_sampled: Number of candidates to randomly sample.
*@li unique: If "unique" is true, samples with rejection,
so that all sampled candidates in a batch are unique. This requires some
approximation to estimate the post-rejection sampling probabilities.
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision. \n

*@par Outputs:
*@li sampled_candidates: A vector of length "num_sampled",
in which each element is the ID of a sampled candidate.
*@li true_expected_count: A "batch_size * num_true" matrix, representing the
number of times each candidate is expected to occur in a batch of sampled candidates.
*If "unique" is true, then this is a probability.
*@li sampled_expected_count: A vector of length "num_sampled", for each
sampled candidate representing the number of times the candidate is expected
to occur in a batch of sampled candidates. If "unique" is true, then this is a probability. \n

*@attention Constraints:
*AllCandidateSampler runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AllCandidateSampler. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
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
*@brief Computes the "ids" of the positions in "sampled_candidates" that
match "true_labels". \n

*@par Inputs:
* @li Input "true_classes" is a 2D matrix.
* @li true_classes: The "true_classes" output of UnpackSparseLabels.
* @li sampled_candidates: The "sampled_candidates" output of CandidateSampler.  \n

*@par Attributes:
*@li num_true: Number of true labels per context.
*@li seed: If either "seed" or "seed2" are set to be non-zero.
*@li seed2: A second seed to avoid seed collision. \n

*@par Outputs:
* @li indices: A vector of indices corresponding to rows of "true_candidates".
* @li ids: A vector of IDs of positions in "sampled_candidates" that match a
"true_label" for the row with the corresponding index in indices.
* @li weights: A vector of the same length as "indices" and "ids", in which
each element is -FLOAT_MAX. \n

*@attention Constraints:
*ComputeAccidentalHits runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ComputeAccidentalHits. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
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

#endif  // OPS_BUILT_IN_OP_PROTO_INC_CANDIDATE_SAMPLING_OPS_H_
