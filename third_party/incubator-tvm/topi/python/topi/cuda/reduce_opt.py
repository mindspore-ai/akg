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
# pylint: disable=invalid-name,unused-variable,too-many-locals,len-as-condition
# 2020.10.10  -  add file for reduction schedule.

"""Schedule for reduce operators"""
from __future__ import absolute_import as _abs
import re
import tvm
from tvm import autotvm
from .. import tag
from .. import generic
from .injective import schedule_injective_from_existing

def _schedule_reduce(op, sch, grid_dims=0, block_dims=0, is_idx_reduce=False, blocksize=[32, 32], autotune=False):
    if autotune:
        cfg = autotvm.get_config()
        cfg.define_knob("tile_x", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        cfg.define_knob("tile_y", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

    if is_idx_reduce:
        data_in = op.input_tensors[0]
        data_out = op.input_tensors[0]
    else:
        data_in = op.input_tensors[0]
        data_out = op.output(0)

    if not sch[data_out].op.reduce_axis:
        return schedule_injective_from_existing(sch, op.output(0))
    
    all_reduce = True
    for i in data_out.shape:
        if i.value != 1:
            all_reduce = False
            break

    if not all_reduce:
        if not autotune:
            num_thread_x = blocksize[0]
            num_thread_y = blocksize[1]
        else:
            num_thread_x = cfg['tile_x'].val
            num_thread_y = cfg['tile_y'].val
        
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
    else:
        num_thread_x = block_dims if block_dims else tvm.target.current_target(allow_none=False).max_num_threads
        thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")

    # Fuse and refactor the reduce axis
    fused_reduce = sch[data_out].fuse(*[sch[data_out].op.reduce_axis[i]
                                        for i in range(len(sch[data_out].op.reduce_axis))])
    
    reduce_axis = list(map(int, re.findall('\d+', str(fused_reduce.var))))
    reduce_with_inner_axis = len(data_in.shape) - 1 in reduce_axis
    
    if all_reduce or reduce_with_inner_axis:
        ko, ki = sch[data_out].split(fused_reduce, factor=num_thread_x)
        reduce_axis_bind_with = thread_x
        if not all_reduce:
            output_axis_bind_with = thread_y
            output_factor = num_thread_y
    else:
        ko, ki = sch[data_out].split(fused_reduce, factor=num_thread_y)
        reduce_axis_bind_with = thread_y
        output_axis_bind_with = thread_x
        output_factor = num_thread_x

    if is_idx_reduce:
        data_out_rf, _ = sch.rfactor(data_out, ki)
    else:
        data_out_rf = sch.rfactor(data_out, ki)

    tx = sch[data_out].op.reduce_axis[0]
    sch[data_out].bind(tx, reduce_axis_bind_with)
    sch[data_out_rf].compute_at(sch[data_out], tx)
    if is_idx_reduce:
        real_output = op.output(0)
        temp_idx_input = data_out.op.output(0)
        temp_val_input = data_out.op.output(1)
    else:
        real_output = data_out
    if not all_reduce:
        # Fuse and split the axis
        fused_outer = sch[real_output].fuse(*[sch[real_output].op.axis[i]
                                              for i in range(len(sch[real_output].op.axis))])
        bx, outer_in = sch[real_output].split(fused_outer, factor=output_factor)

        # Bind the axes to threads and blocks
        sch[real_output].bind(outer_in, output_axis_bind_with)
        sch[real_output].bind(bx, block_x)
        if is_idx_reduce:
            sch[temp_idx_input].compute_at(sch[real_output], outer_in)
            sch[temp_val_input].compute_at(sch[real_output], outer_in)
    else:
        if is_idx_reduce:
            spatial_axis = sch[real_output].fuse(*(sch[real_output].op.axis))
            sch[real_output].bind(spatial_axis, tvm.thread_axis("blockIdx.x"))
            sch[temp_idx_input].compute_at(sch[real_output],
                                           spatial_axis)
            sch[temp_val_input].compute_at(sch[real_output],
                                           spatial_axis)
    sch[real_output].set_store_predicate(reduce_axis_bind_with.equal(0))
    return sch

def traverse_before_reduce(operator, sch, scheduled_ops=None):
    """travserse function"""
    if scheduled_ops == None:
        scheduled_ops = []
    if isinstance(operator, tvm.tensor.PlaceholderOp):
        return scheduled_ops
    if tag.is_injective(operator.tag):
        sch[operator].compute_inline()
        for tensor in operator.input_tensors:
            if tensor.op not in scheduled_ops:
                scheduled_ops = traverse_before_reduce(tensor.op, sch, scheduled_ops)
    elif operator.tag in ("comm_reduce", "comm_reduce_idx"):
        scheduled_ops = traverse_after_reduce(operator, sch, scheduled_ops)
    else:
        raise RuntimeError("Unsupported operator: %s" % operator.tag)

    scheduled_ops.append(operator)
    return scheduled_ops

def traverse_after_reduce(operator, sch, grid_dims = 0, block_dims = 0, scheduled_ops=None, autotune=False):
    """travserse function"""
    if scheduled_ops == None:
        scheduled_ops = []
    if tag.is_injective(operator.tag):
        if operator not in scheduled_ops:
            schedule_injective_from_existing(sch, operator.output(0))
        for tensor in operator.input_tensors:
            scheduled_ops = traverse_after_reduce(tensor.op, sch, scheduled_ops, autotune)
    elif operator.tag == 'comm_reduce':
        _schedule_reduce(operator, sch, grid_dims, block_dims, is_idx_reduce=False, autotune=autotune)
        for tensor in operator.input_tensors:
            if tensor.op not in scheduled_ops:
                scheduled_ops = traverse_before_reduce(tensor.op, sch, scheduled_ops)
    elif operator.tag == 'comm_reduce_idx':
        _schedule_reduce(operator, sch, grid_dims, block_dims, is_idx_reduce=True, autotune=autotune)
        input_tensors = operator.input_tensors[0].op.input_tensors
        for tensor in input_tensors:
            if tensor.op not in scheduled_ops:
                scheduled_ops = traverse_before_reduce(tensor.op, sch, scheduled_ops)
    else:
        raise RuntimeError("Unsupported operator: %s" % operator.tag)

    scheduled_ops.append(operator)
    return scheduled_ops


@generic.schedule_reduce.register(["cuda", "gpu"])
def schedule_reduce(outs, grid_dims = 0, block_dims = 0):
    """Schedule for inject->reduce->bcast ops.

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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    sch = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    for out in outs:
        scheduled_ops = traverse_after_reduce(out.op, sch, grid_dims, block_dims, scheduled_ops)
    return sch

@generic.schedule_reduce.register(["cuda", "gpu"])
def schedule_reduce_autotune(outs):
    """Autotune Schedule for inject->reduce->bcast ops.

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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    sch = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    for out in outs:
        scheduled_ops = traverse_after_reduce(out.op, sch, scheduled_ops=scheduled_ops, autotune=True)
    return sch

