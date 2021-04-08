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
 * \file linalg_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_LINALG_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_LINALG_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Computes the reverse mode backpropagated gradient of the Cholesky
algorithm . \n

*@par Inputs:
*The input x has to be symmetric and positive definite. Inputs include:
*@li x:A Tensor. Must be one of the following types: double, float32. Output
of batch Cholesky algorithm x = cholesky(A). Shape is [..., M, M]. Algorithm
depends only on lower triangular part of the innermost matrices of this tensor.
*@li grad:A Tensor. Must have the same type as l. df/dx where f is some
scalar function. Shape is [..., M, M]. Algorithm depends only on lower
triangular part of the innermost matrices of this tensor . \n

*@par Outputs:
*y:A Tensor. Has the same type as x . \n

*@attention Constraints:
*The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
form square matrices.

*@par Third-party framework compatibility
*Compatible with tensorflow CholeskyGrad operator.
*/

REG_OP(CholeskyGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(CholeskyGrad)

/**
*@brief Computes the Cholesky decomposition of one or more square matrices . \n

*@par Inputs:
*The input x has to be symmetric and positive definite.Inputs include:
*x:A Tensor. Must be one of the following types: double, float32. Shape
is [..., M, M] . \n

*@par Outputs:
*y:A Tensor. Has the same type as x . \n

*@attention Constraints:
*The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
form square matrices.

*@par Third-party framework compatibility
*Compatible with tensorflow Cholesky operator.
*/

REG_OP(Cholesky)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Cholesky)

/**
*@brief Computes the sign and the log of the absolute value of the determinant
of one or more square matrices . \n

*@par Inputs:
*The input x is a tensor of shape [N, M, M] whose inner-most 2 dimensions
form square matrices. Inputs include:
*x:A Tensor. Must be one of the following types: double, float32. Shape is
[..., M, M] . \n

*@par Outputs:
*@li y:A Tensor. Has the same type as x.
*@li sign:A Tensor. Has the same type as x . \n

*@attention Constraints:
*The input x is a tensor of shape [N, M, M] whose inner-most 2 dimensions
form square matrices. \n

*@par Third-party framework compatibility
*Compatible with tensorflow LogMatrixDeterminant operator.
*/

REG_OP(LogMatrixDeterminant)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(sign, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(LogMatrixDeterminant)

/**
*@brief Computes the determinant of one or more square matrices . \n

*@par Inputs:
*The input x is a tensor of shape [N, M, M] whose inner-most 2 dimensions
form square matrices. Inputs include:
*x:A Tensor. Must be one of the following types: double, float32. Shape is
[..., M, M] . \n

*@par Outputs:
*y:A Tensor. Has the same type as x . \n

*@attention Constraints:
*The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
form square matrices.

*@par Third-party framework compatibility
*Compatible with tensorflow MatrixDeterminant operator.
*/

REG_OP(MatrixDeterminant)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(MatrixDeterminant)

/**
*@brief Computes the inverse of one or more square invertible matrices or
their adjoints (conjugate transposes) . \n

*@par Inputs:
*The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
form square matrices. Inputs include:
*x:A Tensor. Must be one of the following types: double, float. Shape is
[..., M, M] . \n

*@par Attributes:
*adjoint:An optional bool. Defaults to False.Boolean indicating whether to
deal with matrix or its (block-wise) adjoint . \n

*@par Outputs:
*y:A Tensor. Has the same type as x . \n

*@attention Constraints:
*The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
form square matrices.  \n

*@par Third-party framework compatibility
*Compatible with tensorflow MatrixInverse operator.
*/

REG_OP(MatrixInverse)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(adjoint, Bool, false)
    .OP_END_FACTORY_REG(MatrixInverse)

/**
*@brief Solves systems of linear equations . \n

*@par Inputs:
*The input rhs must have the same type as matrix. Inputs include:
*@li matrix:A Tensor. Must be one of the following types: double, float.
Shape is [..., M, M].
*@li rhs:A Tensor. Must have the same type as matrix. Shape is [..., M, K] . \n

*@par Attributes:
*adjoint:An optional bool. Defaults to False.Boolean indicating whether to
solve with matrix or its (block-wise) adjoint . \n

*@par Outputs:
*y:A Tensor. Has the same type as matrix . \n

*@attention Constraints:
*The input matrix is a tensor of shape [..., M, M] whose inner-most 2
dimensions form square matrices.  \n

*@par Third-party framework compatibility
*Compatible with tensorflow MatrixSolve operator.
*/

REG_OP(MatrixSolve)
    .INPUT(matrix, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rhs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(adjoint, Bool, false)
    .OP_END_FACTORY_REG(MatrixSolve)

/**
*@brief Solves systems of linear equations . \n

*@par Inputs:
*The input rhs must have the same type as matrix. Inputs include:
*@li matrix:A Tensor. Shape is [..., M, M].
*@li rhs:A Tensor. Must have the same type as matrix. Shape is [..., M, K].
*@li l2:0-D double Tensor. Ignored if fast=False . \n

*@par Attributes:
*fast:bool. Defaults to True . \n

*@par Outputs:
*y:Tensor of shape [..., N, K] whose inner-most 2 dimensions form M-by-K
matrices that solve the equations matrix[..., :, :] * output[..., :, :] =
rhs[..., :, :] in the least squares sense . \n

*@attention Constraints:
*The input matrix matrix is a tensor of shape [..., M, M] whose inner-most 2
dimensions form square matrices.  \n

*@par Third-party framework compatibility
*Compatible with tensorflow MatrixSolveLs operator.
*/

REG_OP(MatrixSolveLs)
    .INPUT(matrix, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rhs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(l2, TensorType({DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(fast, Bool, true)
    .OP_END_FACTORY_REG(MatrixSolveLs)

/**
*@brief Solves systems of linear equations with upper or lower triangular
matrices by backsubstitution . \n

*@par Inputs:
*The input rhs must have the same type as matrix. Inputs include:
*@li matrix: A Tensor. Must be one of the following types: double, float.
Shape is [..., M, M].
*@li rhs:A Tensor. Must have the same type as matrix. Shape is [..., M, K] . \n

*@par Attributes:
*@li lower: An optional bool. Defaults to True. Boolean indicating whether
the innermost matrices in matrix are lower or upper triangular.
*@li An optional bool. Defaults to False. Boolean indicating whether to solve
with matrix or its (block-wise) adjoint . \n

*@par Outputs:
*y:A Tensor. Has the same type as matrix . \n

*@attention Constraints:
*The input matrix is a tensor of shape [..., M, M] whose inner-most 2
dimensions form square matrices.  \n

*@par Third-party framework compatibility
*Compatible with tensorflow MatrixTriangularSolve operator.
*/

REG_OP(MatrixTriangularSolve)
    .INPUT(matrix, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rhs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(lower, Bool, true)
    .ATTR(adjoint, Bool, false)
    .OP_END_FACTORY_REG(MatrixTriangularSolve)

/**
*@brief Computes the QR decompositions of one or more matrices . \n

*@par Inputs:
*The input shape of x must be [..., M, N]. Inputs include:
*x:A Tensor whose shape is [..., M, N]. Must be one of the following types:
double, float . \n

*@par Attributes:
*full_matrices: An optional bool. Defaults to False. If true, compute
full-sized q and r. If false (the default), compute only the leading P
columns of q . \n

*@par Outputs:
*@li q: A Tensor. Has the same type as x.
*@li r: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The input matrix x is a tensor of shape [..., M, N] whose inner-most 2
dimensions form matrices of size [M, N].  \n

*@par Third-party framework compatibility
*Compatible with tensorflow Qr operator.
*/

REG_OP(Qr)
    .INPUT(x, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(q, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(r, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .ATTR(full_matrices, Bool, false)
    .OP_END_FACTORY_REG(Qr)

/**
*@brief Computes the eigen decomposition of a batch of self-adjoint matrices . \n

*@par Inputs:
*The input shape of x must be [..., N, N]. Inputs include:
*x:Tensor of shape [..., N, N]. Only the lower triangular part of each inner
inner matrix is referenced . \n

*@par Attributes:
*compute_v:bool. Defaults to True . \n

*@par Outputs:
*@li eigen_value:Eigenvalues. Shape is [..., N]. Sorted in non-decreasing order.
*@li eigen_vector:Shape is [..., N, N]. The columns of the inner most matrices
contain eigenvectors of the corresponding matrices in tensor

*@attention Constraints:
*The input x is a tensor of shape [..., N, N] whose inner-most 2 dimensions
form square matrices.   \n

*@par Third-party framework compatibility
*Compatible with tensorflow SelfAdjointEig operator.
*/

REG_OP(SelfAdjointEig)
    .INPUT(x, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(eigen_value, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(eigen_vector, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .ATTR(compute_v, Bool, true)
    .OP_END_FACTORY_REG(SelfAdjointEig)

/**
*@brief Computes the singular value decompositions of one or more matrices . \n

*@par Inputs:
*The input shape of x must be [..., N, N]. Inputs include:
*x:Tensor of shape [..., M, N]. Let P be the minimum of M and N . \n

*@par Attributes:
*compute_uv:If True then left and right singular vectors will be computed and
returned in u and v, respectively. Otherwise, only the singular values will
be computed, which can be significantly faster . \n

*@par Outputs:
*@li sigma:Singular values. Shape is [..., P]. The values are sorted in
reverse order of magnitude, so s[..., 0] is the largest value, s[..., 1]
is the second largest, etc.
*@li u:Left singular vectors. If full_matrices is False (default) then shape
is [..., M, P]; if full_matrices is True then shape is [..., M, M]. Not
returned if compute_uv is False.
*@li v:Right singular vectors. If full_matrices is False (default) then shape
is [..., N, P]. If full_matrices is True then shape is [..., N, N]. Not
returned if compute_uv is False . \n

*@attention Constraints:
*The input x is a tensor of shape [..., N, N] whose inner-most 2 dimensions
form square matrices.  \n

*@par Third-party framework compatibility
*Compatible with tensorflow Svd operator
*/

REG_OP(Svd)
    .INPUT(x, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(sigma, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(u, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(v, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .ATTR(compute_uv, Bool, true)
    .ATTR(full_matrices, Bool, false)
    .OP_END_FACTORY_REG(Svd)

/**
*@brief Computes the LU decomposition of one or more square matrices . \n

*@par Inputs:
*input: A tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
matrices of size `[M, M]` . \n

*@par Outputs:
*@li lu: A tensor of shape `[..., M, M]` whose strictly lower triangular part
denotes the lower triangular factor `L` with unit diagonal.
*@li p: upper triangular part denotes the upper triangular factor `U`.Permutation
of the rows encoded as a list of indices in `0..M-1`. Shape is `[..., M]` . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow Lu operator.
*/

REG_OP(Lu)
    .INPUT(input, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(lu, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(p, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(output_idx_type, Type)
    .OP_END_FACTORY_REG(Lu)

/**
*@brief Computes the matrix square root of one or more square matrices . \n

*@par Inputs:
*input: Shape is `[..., M, M]` . \n

*@par Outputs:
y: Shape is `[..., M, M]` . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow MatrixSquareRoot operator.
*/

REG_OP(MatrixSquareRoot)
    .INPUT(input, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(MatrixSquareRoot)

/**
*@brief Solves tridiagonal systems of equations . \n

*@par Inputs:
*@li diagonals: Tensor of shape `[..., 3, M]` whose innermost 2 dimensions represent the tridiagonal matrices with three rows being the superdiagonal, diagonals, and subdiagonals, in order. The last element of the superdiagonal and the first element of the subdiagonal is ignored.
*@li rhs: Tensor of shape `[..., M, K]`, representing K right-hand sides per each
left-hand side . \n

*@par Outputs:
y: Tensor of shape `[..., M, K]` containing the solutions \n

*@par Third-party framework compatibility
* Compatible with TensorFlow TridiagonalSolve operator.
*/

REG_OP(TridiagonalSolve)
    .INPUT(diagonals, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rhs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(partial_pivoting, Bool, true)
    .OP_END_FACTORY_REG(TridiagonalSolve)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_LINALG_OPS_H_
