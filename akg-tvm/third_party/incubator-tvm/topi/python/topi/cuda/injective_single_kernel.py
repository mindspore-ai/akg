# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable,

# 2020.8.27 - Add file injective_single_kernel.py.

"""Schedule for composition of injective operator and generate single cuda kernel."""
from collections import Counter
import tvm
import topi
from tvm import autotvm
from .. import generic, util

@generic.schedule_injective_from_existing.register(["cuda", "gpu"])
def schedule_injective_from_existing(sch, out, tmp_out, fork_node, fake_out, grid_dims = 0, block_dims = 0, autotune=False, buffer_stitch=False):
    """Schedule for injective op from existing schedule.

    Parameters
    ----------
    sch: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the injective op.
    tmp_out: List of Tensor
         The tensors which would be output and as intermediate results in computation.

    fork_node: List of Tensor
         The tensors which are fork nodes in computation.

    fake_out: bool.
         Indicate whether the out tensor is fake or not.

    Returns
    -------
    sch: Schedule
         The updated schedule.
    """
    fused = sch[out].fuse(*sch[out].op.axis)
    kernel_scope, fused = sch[out].split(fused, nparts=1)
    if autotune:
        cfg = autotvm.get_config()
        cfg.define_knob("tile_x", [4, 8, 16, 32, 64, 128, 256, 512, 1024])
        num_thread = cfg['tile_x'].val
        max_block = int(256 * 1024 / num_thread)
    else:
        num_thread = block_dims if block_dims else tvm.target.current_target(allow_none=False).max_num_threads
        max_block = grid_dims if grid_dims else 256

    try:
        const_size = util.get_const_int(util.prod(out.shape))
        need_block_split = const_size > max_block * num_thread
        num_per_thread = (const_size - 1) // (max_block * num_thread) + 1
    except ValueError:
        need_block_split = False

    if need_block_split:
        if not buffer_stitch:
            xo, xi = sch[out].split(fused, factor=num_thread * max_block)
            bx, tx = sch[out].split(xi, factor=num_thread)
            sch[out].reorder(bx, tx, xo)
            inner_most = xo

        else:
            bx, tx = sch[out].split(fused, nparts=max_block)
            xo, tx = sch[out].split(tx, nparts=num_per_thread)
            inner_most = xo
        sch[out].bind(bx, tvm.thread_axis("blockIdx.x"))
        sch[out].bind(tx, tvm.thread_axis("threadIdx.x"))

    else:
        bx, tx = sch[out].split(fused, factor=num_thread)
        inner_most = tx
        sch[out].bind(tx, tvm.thread_axis("threadIdx.x"))
        sch[out].bind(bx, tvm.thread_axis("blockIdx.x"))
    if fake_out:
        sch[out].pragma(kernel_scope, "fake_node", out.name)

    if fork_node:
        for op in fork_node:
            loc_op = sch.cache_write(op, "local")
            sch[loc_op].compute_at(sch[out], inner_most)

    if tmp_out:
        for op in tmp_out:
            sch[op].compute_at(sch[out], inner_most)

    return sch

def create_compute_graph(roots):
    """create compute graph.

    Parameters
    ----------
    roots: List of Tensor

    Returns
    -------
    gmap: Dict of ops and their inputs.
    """
    gmap = {}
    stack = []
    visited = []
    for op in roots:
        stack.append(op)
        visited.append(op)

    while (stack):
        cur_op = stack.pop()
        deps = cur_op.op.input_tensors
        if deps:
            gmap[cur_op] = deps
        for in_op in deps:
            if in_op not in visited:
                visited.append(in_op)
                stack.append(in_op)
    return gmap

def check_multi_out(outs):
    """Detect the leaf nodes in computation graph.

    Parameters
    ----------
    outs: Array of Tensor

    Returns
    -------
    leaf_list: List of Tensor.
        The list of leaf node.
    """
    graph_map = create_compute_graph(outs)
    res_list = list(graph_map.values())
    res = []
    leaf_list = []
    for item in res_list:
        res.extend(item)

    res_count = Counter(res)
    res_key = list(res_count.keys())
    fork_node = []
    for ops, count in res_count.items():
        if count > 1 and (isinstance(ops.op, tvm.tensor.ComputeOp) or (isinstance(ops.op, tvm.tensor.TensorComputeOp))):
            fork_node.append(ops)

    for out in outs:
        if out not in res_key:
            leaf_list.append(out)
    return leaf_list, fork_node

def pick_single_out(outs):
    type_level = {"bool": 1, "int8": 2, "int32": 3, "float16": 4, "float32": 5}
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    ori_out = outs
    fake_out = False
    fork_node = []

    if len(outs) > 1:
        outs, fork_node = check_multi_out(outs)
        if len(outs) > 1:
            fake_op = outs[0]
            highest_type = outs[0].dtype
            for node in outs[1:]:
                if node.dtype != highest_type:
                    if type_level[highest_type] > type_level[node.dtype]:
                        node = topi.cast(node, highest_type)
                    else:
                        highest_type = node.dtype
                        fake_op = topi.cast(fake_op, highest_type)
                fake_op = topi.add(node, fake_op)
                fake_out = True
            outs = [fake_op]
    tmp_out = [op for op in ori_out if op not in outs]
    return outs, tmp_out, fake_out, fork_node


@generic.schedule_injective.register(["cuda", "gpu"])
def schedule_injective(outs, grid_dims=0, block_dims=0, buffer_stitch=False):
    """Schedule for injective op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs, tmp_out, fake_out, fork_node = pick_single_out(outs)
    s = tvm.create_schedule(outs[0].op)

    tvm.schedule.AutoInlineInjective(s)
    schedule_injective_from_existing(s, outs[0], tmp_out, fork_node, fake_out, grid_dims, block_dims, buffer_stitch=buffer_stitch)
    return s

@generic.schedule_injective.register(["cuda", "gpu"])
def schedule_injective_autotune(outs):
    """Schedule for injective op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs, tmp_out, fake_out, fork_node = pick_single_out(outs)
    s = tvm.create_schedule(outs[0].op)

    tvm.schedule.AutoInlineInjective(s)
    schedule_injective_from_existing(s, outs[0], tmp_out, fork_node, fake_out, autotune=True)
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
