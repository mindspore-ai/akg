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

#ifndef GE_OP_STRING_OPS_H_
#define GE_OP_STRING_OPS_H_

#include <sstream>
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Split elements of input based on delimiter into a SparseTensor.

*@par Inputs:
include: \n
*@li input:1-D. Strings to split.
*@li delimiter:0-D. Delimiter characters (bytes), or empty string.

*@par Attributes:
* skip_empty:A bool. If True, skip the empty strings from the result.

*@par Outputs:
*@li indices:A dense matrix of int64 representing the indices of the sparse tensor.
*@li values:A vector of strings corresponding to the splited values.
*@li shape:A length-2 vector of int64 representing the shape of the sparse tensor,\n
*where the first value is N and the second value is the maximum number of tokens\n
*in a single input entry.

*@see StringSplit()

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
*@brief Split elements of source based on sep into a SparseTensor.

*@par Inputs:
include: \n
*@li input:1-D. Strings to split.
*@li sep:0-D string Tensor, the delimiter character.

*@par Attributes:
* maxsplit:An int. If maxsplit > 0, limit of the split of the result.

*@par Outputs:
*@li indices:A dense matrix of int64 representing the indices of the sparse tensor.
*@li values:A vector of strings corresponding to the splited values.
*@li shape:A length-2 vector of int64 representing the shape of the sparse tensor,\n
*where the first value is N and the second value is the maximum number of tokens\n
*in a single input entry.

*@see StringSplitV2()

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
*@brief Determine the script codes of a given tensor of Unicode integer code points.

*@par Inputs:
include: \n
*x:A Tensor of int32 Unicode code points.

*@par Outputs:
*y:A Tensor of int32 script codes corresponding to each input code point.

*@attention Constraints:\n
*This operation converts Unicode code points to script codes corresponding to\n
*each code point.\nScript codes correspond to International Components for\n
*Unicode (ICU) UScriptCode values.\n
*See http://icu-project.org/apiref/icu4c/uscript_8h.html.\n
*Returns -1 (USCRIPT_INVALID_CODE) for invalid codepoints.\n
*Output shape will match input shape.

*@see UnicodeScript()

*/
REG_OP(UnicodeScript)
    .INPUT(x, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(UnicodeScript)

/**
*@brief Return substrings from Tensor of strings.

*@par Inputs:
include: \n
*@li input:Tensor of strings.
*@li pos:Scalar defining the position of first character in each substring.
*@li len:Scalar defining the number of characters to include in each substring.

*@par Outputs:
*output:Tensor of substrings.

*@attention Constraints:\n
*The hash function is deterministic on the content of the string within\n
*the process and will never change. However, it is not suitable for\n
*cryptography. This function may be used when CPU time is scarce and\n
*inputs are trusted or unimportant. There is a risk of adversaries\n
*constructing inputs that all hash to the same bucket.\n
*To prevent this problem, use a strong hash function with\n

*@see Substr()

*/
REG_OP(Substr)
    .INPUT(input, TensorType({DT_STRING}))
    .INPUT(pos, TensorType({DT_INT32, DT_INT64}))
    .INPUT(len, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(output, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(Substr)

/**
*@brief Converts each string in the input Tensor to its hash mod by a number of buckets.

*@par Inputs:
include: \n
*string_tensor:The strings to assign a hash bucket.

*@par Outputs:
*y:A Tensor of the same shape as the input x.

*@attention Constraints:\n
*The hash function is deterministic on the content of the string within\n
*the process and will never change. However, it is not suitable for cryptography.\n
*This function may be used when CPU time is scarce and inputs are trusted or\n
*unimportant. There is a risk of adversaries constructing inputs that all hash\n
*to the same bucket. To prevent this problem, use a strong hash function with\n

*@see StringToHashBucketFast()

*/
REG_OP(StringToHashBucketFast)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .OP_END_FACTORY_REG(StringToHashBucketFast)

/**
*@brief Converts each string in the input Tensor to its hash mod by a number of buckets.

*@par Inputs:
include: \n
*x:The strings to assign a hash bucket.

*@par Attributes:
*num_buckets:The number of buckets.

*@par Outputs:
*y:A Tensor of the same shape as the input x.

*@attention Constraints:\n
*@li A strong hash is important when inputs may be malicious, e.g. URLs with\n
*additional components. Adversaries could try to make their inputs hash to\n
*the same bucket for a denial-of-service attack or to skew the results.\n
*A strong hash can be used to make it difficult to find inputs with a skewed\n
* hash value distribution over buckets. This requires that the hash function\
*is seeded by a high-entropy (random) "key" unknown to the adversary.
*@li The additional robustness comes at a cost of roughly 4x higher\n

*@see StringToHashBucketStrong()

*/
REG_OP(StringToHashBucketStrong)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .REQUIRED_ATTR(key, ListInt)
    .OP_END_FACTORY_REG(StringToHashBucketStrong)

/**
*@brief Converts each string in the input Tensor to its hash mod by a number of buckets.

*@par Inputs:
include: \n
*string_tensor:The strings to assign a hash bucket.

*@par Attributes:
*num_buckets:The number of buckets.

*@par Outputs:
*y:A Tensor of the same shape as the input string_tensor.

*@see StringToHashBucket()

*/
REG_OP(StringToHashBucket)
    .INPUT(string_tensor, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(num_buckets, Int, 1)
    .OP_END_FACTORY_REG(StringToHashBucket)

/**
*@brief Strip leading and trailing whitespaces from the Tensor.

*@par Inputs:
include: \n
*x:A string Tensor of any shape.

*@par Outputs:
*y:A string Tensor of the same shape as the input.

*@see StringStrip()

*/
REG_OP(StringStrip)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(StringStrip)

/**
*@brief Computes the length of each string given in the input tensor.

*@par Inputs:
include: \n
*x:The string for which to compute the length.

*@par Attributes:
*unit:The unit that is counted to compute string length.\n
*One of: "BYTE" (for the number of bytes in each string) or\n
*"UTF8_CHAR" (for the number of UTF-8 encoded Unicode code points in each string).\n
*Results are undefined if unit=UTF8_CHAR and the input strings do not contain\N
*structurally valid UTF-8.

*@par Outputs:
*y:Integer tensor that has the same shape as input.\n
*The output contains the element-wise string lengths of input.

*@see StringLength()

*/
REG_OP(StringLength)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(unit, String, "BYTE")
    .OP_END_FACTORY_REG(StringLength)

/**
*@brief Joins the strings in the given list of string tensors into one tensor.

*@par Inputs:
*The input is a string tensor of any shape. The pattern is a scalar string tensor\n
*which is applied to every element of the input tensor. The boolean values\n
*(True or False) of the output tensor indicate if the input matches the regex\n
*pattern provided. The pattern follows the re2 syntax\n
*(https://github.com/google/re2/wiki/Syntax).: \n
include: \n
*x:A list of string tensors. The tensors must all have the same shape,\n
*or be scalars. Scalars may be mixed in; these will be broadcast to the shape\n
*of non-scalar inputs.

*@par Attributes:
*@li N:The length of input x.
*@li separator:string, an optional join separator.

*@par Outputs:
*y:The output tensor.

*@see StringJoin()

*/
REG_OP(StringJoin)
    .DYNAMIC_INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .REQUIRED_ATTR(N, Int)
    .ATTR(separator, String, "")
    .OP_END_FACTORY_REG(StringJoin)

/**
*@brief Formats a string template using a list of tensors.

*@par Inputs:
*The input is a string tensor of any shape. The pattern is a scalar string tensor\n
*which is applied to every element of the input tensor.\n
*The boolean values (True or False) of the output tensor indicate if the input\n
*matches the regex pattern provided. The pattern follows the re2 syntax\n
*(https://github.com/google/re2/wiki/Syntax).: \n
include: \n
*x:The tensors to format into the placeholder string.

*@par Attributes:
*@li template:A string, the template to format tensor summaries into.
*@li placeholder:A string, at each placeholder in the template a subsequent tensor summary will be inserted.
*@li summarize:When formatting the tensor summaries print the first and last summarize entries of each tensor dimension.

*@par Outputs:
*y:The resulting string scalar.

*@see StringFormat()

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
*@brief Check if the input matches the regex pattern.

*@par Inputs:
*The input is a string tensor of any shape. The pattern is a scalar string tensor\n
*which is applied to every element of the input tensor. The boolean values \n
*(True or False) of the output tensor indicate if the input matches the regex\n
*pattern provided. The pattern follows the re2 syntax\n
*(https://github.com/google/re2/wiki/Syntax).: \n
include: \n
*@li x:A string tensor of the text to be processed.
*@li pattern:A scalar string tensor containing the regular expression to match the input.

*@par Outputs:
*y:A bool tensor with the same shape as input.

*@see RegexFullMatch()

*/
REG_OP(RegexFullMatch)
    .INPUT(x, TensorType({DT_STRING}))
    .INPUT(pattern, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(RegexFullMatch)

/**
*@brief Replaces matches of the pattern regular expression in input with the\n
*replacement string provided in rewrite.

*@par Inputs:
*It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax).: \n
include: \n
*@li x:The text to be processed.
*@li pattern:The regular expression to be matched in the input strings.
*@li rewrite:The rewrite string to be substituted for the pattern expression\n
*where it is matched in the input strings.

*@par Attributes:
*replace_global:If True, the replacement is global\n
*(that is, all matches of the pattern regular expression in each input string\n
*are rewritten), otherwise the rewrite substitution is only made for the first\n
* pattern match.

*@par Outputs:
*y:The text after applying pattern match and rewrite substitution.

*@see RegexReplace()

*/
REG_OP(RegexReplace)
    .INPUT(x, TensorType({DT_STRING}))
    .INPUT(pattern, TensorType({DT_STRING}))
    .INPUT(rewrite, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(replace_global, Bool, true)
    .OP_END_FACTORY_REG(RegexReplace)

/**
*@brief Converts each entry in the given tensor to strings.

*@par Inputs:
*Supports many numeric types and boolean.: \n
include: \n
*x:A tensor can be trans to string.

*@par Attributes:
*@li precision:The post-decimal precision to use for floating point numbers.\n
*Only used if precision > -1.
*@li scientific:Use scientific notation for floating point numbers.
*@li shortest:Use shortest representation (either scientific or standard)\n
*for floating point numbers..
*@li width:Pad pre-decimal numbers to this width. Applies to both floating\n
*point and integer numbers. Only used if width > -1.
*@li fill:The value to pad if width > -1. If empty, pads with spaces.\n
*Another typical value is '0'. String cannot be longer than 1 character.

*@par Outputs:
*y:The output tensor.

*@see AsString()

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
*@brief Encode strings into web-safe base64 format.

*@par Inputs:
*Input may or may not have padding at the end. See EncodeBase64 for padding.\n
*Web-safe means that input must use - and _ instead of + and /.: \n
include: \n
*x:Strings to be encoded.

*@par Attributes:
*pad:Bool whether padding is applied at the ends.

*@par Outputs:
*y:Input strings encoded in base64.

*@attention Constraints:\n
*Refer to the following article for more information on base64 format:\n
*en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '='\n
*at the end so that the encoded has length multiple of 4.\n
*See Padding section of the link above. Web-safe means that the encoder\n
*uses - and _ instead of + and /.

*@see EncodeBase64()

*/
REG_OP(EncodeBase64)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .ATTR(pad, Bool, false)
    .OP_END_FACTORY_REG(EncodeBase64)

/**
*@brief Decode web-safe base64-encoded strings.

*@par Inputs:
*Input may or may not have padding at the end. See EncodeBase64 for padding.\n
*Web-safe means that input must use - and _ instead of + and /.: \n
include: \n
*x:Base64 strings to decode.

*@par Outputs:
*y:Decoded strings.

*@see DecodeBase64()

*/
REG_OP(DecodeBase64)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(DecodeBase64)
}  // namespace ge

#endif  // GE_OP_STRING_OPS_H_
