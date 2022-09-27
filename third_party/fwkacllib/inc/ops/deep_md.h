/**
 * CCopyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file deep_md.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_DEEP_MD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_DEEP_MD_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Calculate TabulateFusion. \n
*
* @par Inputs:
* Five inputs, including:
* @li table: A Tensor. Must be one of the following types: float16, float32, float64.
* @li table_info: A Tensor. Must be one of the following types: float16, float32, float64.
* @li em_x: A Tensor. Must be one of the following types: float16, float32, float64.
* @li em: A Tensor. Must be one of the following types: float16, float32, float64. \n
*
* @par Outputs:
* descriptor: A Tensor. Must be one of the following types: float16, float32, float64. \n
*
* @par Attributes:
* Three attributes, including:
* @li last_layer_size: int value.
* @li split_count: int value.
* @li split_index: int value. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TabulateFusion)
    .INPUT(table, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(table_info, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(em_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(em, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(descriptor, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(last_layer_size, Int)
    .OP_END_FACTORY_REG(TabulateFusion)

/**
* @brief Calculate ProdEnvMatA. \n
*
* @par Inputs:
* @li coord: A Tensor. Must be one of the following types: float32, float64.
* @li type: A Tensor. Must be one of the following types: int32.
* @li natoms: A Tensor. Must be one of the following types: int32.
* @li box: A Tensor. Must be one of the following types: float32, float64.
* @li mesh: A Tensor. Must be one of the following types: int32.
* @li davg: A Tensor. Must be one of the following types: float32, float64.
* @li dstd: A Tensor. Must be one of the following types: float32, float64.
*
* @par Outputs:
* descrpt: A Tensor. Must be one of the following types: float32, float64.
* descrpt_deriv: A Tensor. Must be one of the following types: float32, float64.
* rij: A Tensor. Must be one of the following types: float32, float64.
* nlist: A Tensor. Must be one of the following types: int32. \n
*
* @par Attributes:
* @li rcut_a: A Float.
* @li rcut_r: A Float.
* @li rcut_r_smth: A Float.
* @li sel_a: A ListInt.
* @li split_count: A Int.
* @li split_index: A Int.\n
*
*/
REG_OP(ProdEnvMatA)
    .INPUT(coord, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(type, TensorType({DT_INT32}))
    .INPUT(natoms, TensorType({DT_INT32}))
    .INPUT(box, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(mesh, TensorType({DT_INT32}))
    .INPUT(davg, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(dstd, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(descrpt, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(descrpt_deriv, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(rij, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(nlist, TensorType({DT_INT32}))
    .ATTR(rcut_a, Float, 1.0)
    .ATTR(rcut_r, Float, 1.0)
    .ATTR(rcut_r_smth, Float, 1.0)
    .ATTR(sel_a, ListInt, {})
    .ATTR(sel_r, ListInt, {})
    .OP_END_FACTORY_REG(ProdEnvMatA)

/**
* @brief Calculate ProdEnvMatACalRij. 
* Use type, natoms, sel_a, and rcut_r as constraints, find the central element in
* the corresponding coord through mesh, output the index of the central element 
* and the distance between the central element and each neighbor. \n
*
* @par Inputs:
* @li coord: A Tensor. Must be one of the following types: float32, float64.
* @li type: A Tensor. Must be one of the following types: int32.
* @li natoms: A Tensor. Must be one of the following types: int32.
* @li box: A Tensor. Must be one of the following types: float32, float64.
* @li mesh: A Tensor. Must be one of the following types: int32. 
*
* @par Outputs:
* rij: A Tensor. Must be one of the following types: float32, float64.
* nlist: A Tensor. Must be one of the following types: int32.
* distance: A Tensor. Must be one of the following types: float32, float64.
* rij_x: A Tensor. Must be one of the following types: float32, float64.
* rij_y: A Tensor. Must be one of the following types: float32, float64.
* rij_z: A Tensor. Must be one of the following types: float32, float64. \n
*
* @par Attributes:
* @li rcut_a: A Float.
* @li rcut_r: A Float.
* @li rcut_r_smth: A Float.
* @li sel_a: A ListInt.
* @li sel_r: A ListInt. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ProdEnvMatACalcRij)
    .INPUT(coord, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(type, TensorType({DT_INT32}))
    .INPUT(natoms, TensorType({DT_INT32}))
    .INPUT(box, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(mesh, TensorType({DT_INT32}))
    .OUTPUT(rij, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(nlist, TensorType({DT_INT32}))
    .OUTPUT(distance, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(rij_x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(rij_y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(rij_z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(rcut_a, Float, 1.0)
    .ATTR(rcut_r, Float, 1.0)
    .ATTR(rcut_r_smth, Float, 1.0)
    .ATTR(sel_a, ListInt, {})
    .ATTR(sel_r, ListInt, {})
    .OP_END_FACTORY_REG(ProdEnvMatACalcRij)

/**
* @brief Calculate ProdEnvMatACalcDescrpt. \n
*
* @par Inputs:
* @li distance: A Tensor. Must be one of the following types: float32, float64.
* @li rij_x: A Tensor. Must be one of the following types: float32, float64.
* @li rij_y: A Tensor. Must be one of the following types: float32, float64.
* @li rij_z: A Tensor. Must be one of the following types: float32, float64.
* @li type: A Tensor. Must be one of the following types: int32.
* @li natoms: A Tensor. Must be one of the following types: int32.
* @li mesh: A Tensor. Must be one of the following types: int32.
* @li davg: A Tensor. Must be one of the following types: float32, float64.
* @li dstd: A Tensor. Must be one of the following types: float32, float64. \n
*
* @par Outputs:
* @li descrpt: A Tensor. Must be one of the following types: float32, float64.
* @li descrpt_deriv: A Tensor. Must be one of the following types: float32, float64. \n
*
* @par Attributes:
* @li rcut_a: A Float.
* @li rcut_r: A Float.
* @li rcut_r_smth: A Float.
* @li sel_a: A ListInt.
* @li sel_r: A ListInt. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ProdEnvMatACalcDescrpt)
    .INPUT(distance, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rij_x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rij_y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rij_z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(type, TensorType({DT_INT32}))
    .INPUT(natoms, TensorType({DT_INT32}))
    .INPUT(mesh, TensorType({DT_INT32}))
    .INPUT(davg, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(dstd, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(descrpt, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(descrpt_deriv, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(rcut_a, Float, 1.0)
    .ATTR(rcut_r, Float, 1.0)
    .ATTR(rcut_r_smth, Float, 1.0)
    .ATTR(sel_a, ListInt, {})
    .ATTR(sel_r, ListInt, {})
    .OP_END_FACTORY_REG(ProdEnvMatACalcDescrpt)

/**
* @brief Calculate ProdForceSeA. \n
*
* @par Inputs:
* Five inputs, including:
* @li net_deriv: A Tensor. Must be one of the following types: float16, float32, float64.
* @li in_deriv: A Tensor. Must be one of the following types: float16, float32, float64.
* @li nlist: A Tensor. dtype is int32.
* @li natoms: A Tensor. dtype is int32. \n
*
* @par Outputs:
* atom_force: A Tensor. Must be one of the following types: float16, float32, float64. \n
*
* @par Attributes:
* Two attributes, including:
* @li n_a_sel: A Scalar.
* @li n_r_sel: A Scalar. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ProdForceSeA)
    .INPUT(net_deriv, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(in_deriv, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(nlist, TensorType({DT_INT32}))
    .INPUT(natoms, TensorType({DT_INT32}))
    .OUTPUT(atom_force, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(n_a_sel, Int)
    .REQUIRED_ATTR(n_r_sel, Int)
    .OP_END_FACTORY_REG(ProdForceSeA)

/**
* @brief Calculate ProdVirialSeA. \n
*
* @par Inputs:
* Five inputs, including:
* @li net_deriv: A Tensor. Must be one of the following types: float16, float32, float64.
* @li in_deriv: A Tensor. Must be one of the following types: float16, float32, float64.
* @li rij: A Tensor. Must be one of the following types: float16, float32, float64.
* @li nlist: A Tensor. dtype is int32.
* @li natoms: A Tensor. dtype is int32. \n
*
* @par Outputs:
* Two outputs, including:
* @li virial: A Tensor. Must be one of the following types: float16, float32, float64.
* @li atom_virial: A Tensor. Must be one of the following types: float16, float32, float64. \n
*
* @par Attributes:
* Two attributes, including:
* @li n_a_sel: Int value.
* @li n_r_sel: Int value.
* @li split_count: Int value.
* @li split_index: Int value. \n
*/
REG_OP(ProdVirialSeA)
    .INPUT(net_deriv, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(in_deriv, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(rij, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(nlist, TensorType({DT_INT32}))
    .INPUT(natoms, TensorType({DT_INT32}))
    .OUTPUT(virial, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(atom_virial, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(n_a_sel, Int)
    .REQUIRED_ATTR(n_r_sel, Int)
    .OP_END_FACTORY_REG(ProdVirialSeA)

/**
* @brief Calculate TabulateFusionGrad. \n
*
* @par Inputs:
* Five inputs, including:
* @li table: A Tensor. Must be one of the following types: float16, float32, float64.
* @li table_info: A Tensor. Must be one of the following types: float16, float32, float64.
* @li em_x: A Tensor. Must be one of the following types: float16, float32, float64.
* @li em: A Tensor. Must be one of the following types: float16, float32, float64.
* @li dy: A Tensor. Must be one of the following types: float16, float32, float64.
* @li descriptor: A Tensor. Must be one of the following types: float16, float32, float64. \n
*
* @par Outputs:
* @li dy_dem_x: A Tensor. Must be one of the following types: float16, float32, float64.
* @li dy_dem: A Tensor. Must be one of the following types: float16, float32, float64. \n
*
* @par Attributes:
* Two attributes, including:
* @li split_count: A Scalar. 
* @li split_index: A Scalar. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TabulateFusionGrad)
  .INPUT(table, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(table_info, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(em_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(em, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(descriptor, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(dy_dem_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(dy_dem, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .OP_END_FACTORY_REG(TabulateFusionGrad)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_DEEP_MD_H_
