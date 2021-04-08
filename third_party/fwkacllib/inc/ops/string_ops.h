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

/*!
 * \file string_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STRING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STRING_OPS_H_

#include <sstream>
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Split elements of input based on delimiter into a SparseTensor . \n

*@par Inputs:
include:
*@li input:1-D. Strings to split.
*@li delimiter:0-D. Delimiter characters (bytes), or empty string . \n

*@par Attributes:
* skip_empty:A bool. If True, skip the empty strings from the result . \n

*@par Outputs:
*@li indices:A dense matrix of int64 representing the indices of the sparse tensor.
*@li values:A vector of strings corresponding to the splited values.
*@li shape:A length-2 vector of int64 representing the shape of the sparse tensor,
*where the first value is N and the second value is the maximum number of tokens
*in a single input entry . \n

*@see StringSplit()

*@par Third-party framework compatibility
*compatible with StringSplit op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringSplit)
    .INPUT(input, TensorType({DT_STRING}))
    .INPUT(delimiter, TensorType({DT_STRING}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_STRING}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .ATTR(skip_empty, Bool, true)
    .OP_END_FACTORY_REG(StringSplit)

/**
*@brief Split elements of source based on sep into a SparseTensor . \n

*@par Inputs:
include:
*@li input:1-D. Strings to split.
*@li sep:0-D string Tensor, the delimiter character . \n

*@par Attributes:
* maxsplit:An int. If maxsplit > 0, limit of the split of the result . \n

*@par Outputs:
*@li indices:A dense matrix of int64 representing the indices of the sparse tensor.
*@li values:A vector of strings corresponding to the splited values.
*@li shape:A length-2 vector of int64 representing the shape of the sparse tensor,
*where the first value is N and the second value is the maximum number of tokens
*in a single input entry . \n

*@see StringSplitV2()

*@par Third-party framework compatibility
*compatible with StringSplitV2 op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringSplitV2)
    .INPUT(input, TensorType({DT_STRING}))
    .INPUT(sep, TensorType({DT_STRING}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_STRING}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .ATTR(maxsplit, Int, -1)
    .OP_END_FACTORY_REG(StringSplitV2)

/**
*@brief Determine the script codes of a given tensor of Unicode integer code points . \n

*@par Inputs:
include:
*x:A Tensor of int32 Unicode code points . \n

*@par Outputs:
*y:A Tensor of int32 script codes corresponding to each input code point . \n

*@attention Constraints:
*This operation converts Unicode code points to script codes corresponding to
*each code point. Script codes correspond to International Components for
*Unicode (ICU) UScriptCode values.
*See http://icu-project.org/apiref/icu4c/uscript_8h.html.
*Returns -1 (USCRIPT_INVALID_CODE) for invalid codepoints.
*Output shape will match input shape . \n

*@see UnicodeScript()

*@par Third-party framework compatibility
*compatible with UnicodeScript op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(UnicodeScript)
    .INPUT(x, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(UnicodeScript)

/**
*@brief Return substrings from Tensor of strings . \n

*@par Inputs:
include:
*@li input:Tensor of strings.
*@li pos:Scalar defining the position of first character in each substring.
*@li len:Scalar defining the number of characters to include in each substring . \n

*@par Outputs:
*output:Tensor of substrings . \n

*@attention Constraints:
*The hash function is deterministic on the content of the string within
*the process and will never change. However, it is not suitable for
*cryptography. This function may be used when CPU time is scarce and
*inputs are trusted or unimportant. There is a risk of adversaries
*constructing inputs that all hash to the same bucket.
*To prevent this problem, use a strong hash function with
*tf.string_to_hash_bucket_strong . \n

*@see Substr()

*@par Third-party framework compatibility
*compatible with Substr op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Substr)
    .INPUT(input, TensorType({DT_STRING}))
    .INPUT(pos, TensorType({DT_INT32, DT_INT64}))
    .INPUT(len, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(output, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(Substr)

/**
*@brief Converts each string in the input Tensor to its hash mod by a number of buckets . \n

*@par Inputs:
include:
*string_tensor:The strings to assign a hash bucket . \n

*@par Outputs:
*y:A Tensor of the same shape as the input x . \n

*@attention Constraints:
*The hash function is deterministic on the content of the string within
*the process and will never change. However, it is not suitable for cryptography.
*This function may be used when CPU time is scarce and inputs are trusted or
*unimportant. There is a risk of adversaries constructing inputs that all hash
*to the same bucket. To prevent this problem, use a strong hash function with
*tf.string_to_hash_bucket_strong . \n

*@see StringToHashBucketFast()

*@par Third-party framework compatibility
*compatible with StringToHashBucketFast op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringToHashBucketFast)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .OP_END_FACTORY_REG(StringToHashBucketFast)

/**
*@brief Converts each string in the input Tensor to its hash mod by a number of buckets . \n

*@par Inputs:
include:
*x:The strings to assign a hash bucket . \n

*@par Attributes:
*num_buckets:The number of buckets . \n

*@par Outputs:
*y:A Tensor of the same shape as the input x . \n

*@attention Constraints:
*@li A strong hash is important when inputs may be malicious, e.g. URLs with
*additional components. Adversaries could try to make their inputs hash to
*the same bucket for a denial-of-service attack or to skew the results.
*A strong hash can be used to make it difficult to find inputs with a skewed
* hash value distribution over buckets. This requires that the hash function\
*is seeded by a high-entropy (random) "key" unknown to the adversary.
*@li The additional robustness comes at a cost of roughly 4x higher
*compute time than tf.string_to_hash_bucket_fast . \n

*@see StringToHashBucketStrong()

*@par Third-party framework compatibility
*compatible with StringToHashBucketStrong op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringToHashBucketStrong)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .REQUIRED_ATTR(key, ListInt)
    .OP_END_FACTORY_REG(StringToHashBucketStrong)

/**
*@brief Converts each string in the input Tensor to its hash mod by a number of buckets . \n

*@par Inputs:
include:
*string_tensor:The strings to assign a hash bucket . \n

*@par Attributes:
*num_buckets:The number of buckets . \n

*@par Outputs:
*y:A Tensor of the same shape as the input string_tensor . \n

*@see StringToHashBucket()

*@par Third-party framework compatibility
*compatible with StringToHashBucket op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringToHashBucket)
    .INPUT(string_tensor, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .OP_END_FACTORY_REG(StringToHashBucket)

/**
*@brief Strip leading and trailing whitespaces from the Tensor . \n

*@par Inputs:
include:
*x:A string Tensor of any shape . \n

*@par Outputs:
*y:A string Tensor of the same shape as the input . \n

*@see StringStrip()

*@par Third-party framework compatibility
*compatible with StringStrip op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringStrip)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(StringStrip)

/**
*@brief Computes the length of each string given in the input tensor . \n

*@par Inputs:
include:
*x:The string for which to compute the length . \n

*@par Attributes:
*unit:The unit that is counted to compute string length.
*One of: "BYTE" (for the number of bytes in each string) or
*"UTF8_CHAR" (for the number of UTF-8 encoded Unicode code points in each string).
*Results are undefined if unit=UTF8_CHAR and the input strings do not contain
*structurally valid UTF-8 . \n

*@par Outputs:
*y:Integer tensor that has the same shape as input.
*The output contains the element-wise string lengths of input . \n

*@see StringLength()

*@par Third-party framework compatibility
*compatible with StringLength op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringLength)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(unit, String, "BYTE")
    .OP_END_FACTORY_REG(StringLength)

/**
*@brief Joins the strings in the given list of string tensors into one tensor . \n

*@par Inputs:
*The input is a string tensor of any shape. The pattern is a scalar string tensor
*which is applied to every element of the input tensor. The boolean values
*(True or False) of the output tensor indicate if the input matches the regex
*pattern provided. The pattern follows the re2 syntax
*(https://github.com/google/re2/wiki/Syntax).:
include:
*x:A list of string tensors. The tensors must all have the same shape,
*or be scalars. Scalars may be mixed in; these will be broadcast to the shape
*of non-scalar inputs . It's a dynamic input. \n

*@par Attributes:
*@li N:The length of input x.
*@li separator:string, an optional join separator . \n

*@par Outputs:
*y:The output tensor . \n

*@see StringJoin()

*@par Third-party framework compatibility
*compatible with StringJoin op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringJoin)
    .DYNAMIC_INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .REQUIRED_ATTR(N, Int)
    .ATTR(separator, String, "")
    .OP_END_FACTORY_REG(StringJoin)

/**
*@brief Formats a string template using a list of tensors . \n

*@par Inputs:
*The input is a string tensor of any shape. The pattern is a scalar string tensor
*which is applied to every element of the input tensor.
*The boolean values (True or False) of the output tensor indicate if the input
*matches the regex pattern provided. The pattern follows the re2 syntax
*(https://github.com/google/re2/wiki/Syntax).:
include:
*x:The tensors to format into the placeholder string . It's a dynamic input. \n

*@par Attributes:
*@li template:A string, the template to format tensor summaries into.
*@li placeholder:A string, at each placeholder in the template a subsequent tensor summary will be inserted.
*@li summarize:When formatting the tensor summaries print the first and last summarize entries of each tensor dimension . \n

*@par Outputs:
*y:The resulting string scalar . \n

*@see StringFormat()

*@par Third-party framework compatibility
* compatible with StringFormat op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(StringFormat)
    .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_STRING, DT_FLOAT16, \
        DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(template, String, "%s")
    .ATTR(placeholder, String, "%s")
    .ATTR(summarize, Int, 3)
    .OP_END_FACTORY_REG(StringFormat)

/**
*@brief Check if the input matches the regex pattern . \n

*@par Inputs:
*The input is a string tensor of any shape. The pattern is a scalar string tensor
*which is applied to every element of the input tensor. The boolean values
*(True or False) of the output tensor indicate if the input matches the regex
*pattern provided. The pattern follows the re2 syntax
*(https://github.com/google/re2/wiki/Syntax).:
include:
*@li x:A string tensor of the text to be processed.
*@li pattern:A scalar string tensor containing the regular expression to match the input . \n

*@par Outputs:
*y:A bool tensor with the same shape as input . \n

*@see RegexFullMatch()

*@par Third-party framework compatibility
*compatible with RegexFullMatch op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(RegexFullMatch)
    .INPUT(x, TensorType({DT_STRING}))
    .INPUT(pattern, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(RegexFullMatch)

/**
*@brief Replaces matches of the pattern regular expression in input with the
*replacement string provided in rewrite . \n

*@par Inputs:
*It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax).:
include:
*@li x:The text to be processed.
*@li pattern:The regular expression to be matched in the input strings.
*@li rewrite:The rewrite string to be substituted for the pattern expression
*where it is matched in the input strings . \n

*@par Attributes:
*replace_global:If True, the replacement is global
*(that is, all matches of the pattern regular expression in each input string
*are rewritten), otherwise the rewrite substitution is only made for the first
* pattern match . \n

*@par Outputs:
*y:The text after applying pattern match and rewrite substitution . \n

*@see RegexReplace()

*@par Third-party framework compatibility
*compatible with RegexReplace op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(RegexReplace)
    .INPUT(x, TensorType({DT_STRING}))
    .INPUT(pattern, TensorType({DT_STRING}))
    .INPUT(rewrite, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(replace_global, Bool, true)
    .OP_END_FACTORY_REG(RegexReplace)

/**
*@brief Converts each entry in the given tensor to strings . \n

*@par Inputs:
*Supports many numeric types and boolean.:
include:
*x:A tensor can be trans to string . \n

*@par Attributes:
*@li precision:The post-decimal precision to use for floating point numbers.
*Only used if precision > -1.
*@li scientific:Use scientific notation for floating point numbers.
*@li shortest:Use shortest representation (either scientific or standard)
*for floating point numbers..
*@li width:Pad pre-decimal numbers to this width. Applies to both floating
*point and integer numbers. Only used if width > -1.
*@li fill:The value to pad if width > -1. If empty, pads with spaces.
*Another typical value is '0'. String cannot be longer than 1 character . \n

*@par Outputs:
*y:The output tensor . \n

*@see AsString()

*@par Third-party framework compatibility
*compatible with AsString op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AsString)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT, \
        DT_DOUBLE, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(precision, Int, -1)
    .ATTR(scientific, Bool, false)
    .ATTR(shortest, Bool, false)
    .ATTR(width, Int, -1)
    .ATTR(fill, String, "")
    .OP_END_FACTORY_REG(AsString)

/**
*@brief Encode strings into web-safe base64 format . \n

*@par Inputs:
*Input may or may not have padding at the end. See EncodeBase64 for padding.
*Web-safe means that input must use - and _ instead of + and /.:
include:
*x:Strings to be encoded . \n

*@par Attributes:
*pad:Bool whether padding is applied at the ends . \n

*@par Outputs:
*y:Input strings encoded in base64 . \n

*@attention Constraints:
*Refer to the following article for more information on base64 format:
*en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '='
*at the end so that the encoded has length multiple of 4.
*See Padding section of the link above. Web-safe means that the encoder
*uses - and _ instead of + and / . \n

*@see EncodeBase64()

*@par Third-party framework compatibility
*compatible with EncodeBase64 op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(EncodeBase64)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(pad, Bool, false)
    .OP_END_FACTORY_REG(EncodeBase64)

/**
*@brief Decode web-safe base64-encoded strings . \n

*@par Inputs:
*Input may or may not have padding at the end. See EncodeBase64 for padding.
*Web-safe means that input must use - and _ instead of + and /.:
include:
*x:Base64 strings to decode . \n

*@par Outputs:
*y:Decoded strings . \n

*@see DecodeBase64()

*@par Third-party framework compatibility
*compatible with DecodeBase64 op of tensorflow

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DecodeBase64)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(DecodeBase64)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_STRING_OPS_H_
