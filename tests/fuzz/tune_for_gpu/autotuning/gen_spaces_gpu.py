from .kernel_compiler import compile_kernel
from collections import namedtuple
from .space import ListConfigSpace

def get_reduce_axis_length(in_shape,reduce_axis):
  lx, ly = 1, 1
  if reduce_axis == None or len(reduce_axis) == len(in_shape):
    for v in in_shape: lx *= v
  elif (len(in_shape) - 1) in reduce_axis:
    for i in range(len(in_shape)):
      if i in reduce_axis: 
        lx *= in_shape[i]
      else:
        ly *= in_shape[i]

  else:
    for i in range(len(in_shape)):
      if i in reduce_axis: 
        ly *= in_shape[i]
      else:
        lx *= in_shape[i]

  return lx, ly
    

def _get_space_reduce_gpu_manually(op_type: str, op_desc, tuning_attrs=[], tuning_attrs_info=None):
  """get config space of reduce_sum operators in gpu"""
  space_res, key, expect, input_for_mod = compile_kernel(op_type, op_desc, None, None, None, 0,
                                                          gen_tiling_spaces=True)
  
  in_shape, reduce_axis = op_desc[2].in_shape, op_desc[2].axis
  dim_len = 1 if reduce_axis == None or len(reduce_axis) == len(in_shape) else 2
  dim_names = ['tiling_' + str(i) for i in range(dim_len)]
  dim_names.append("block_x")
  dim_names.append("block_y")  
  dim_names.append("block_z") 
  dim_names.append("thread_x")
  dim_names.append("thread_y")
  dim_names.append("thread_z")
  for key in tuning_attrs_info[0]:
    dim_names.append(key)
  lx, ly =  get_reduce_axis_length(in_shape, reduce_axis)

  tiling_spaces = []
  if reduce_axis == None or len(reduce_axis) == len(in_shape):
    """all-reduce"""
    possible_tx_list = [2**i for i in range(4,11)]
    for tx in possible_tx_list:
      if tx > lx: break
      possible_dim0_list = [d0 for d0 in range(tx, lx+1, tx)]
      if possible_dim0_list[-1] != lx: possible_dim0_list.append(lx)
      for d0 in possible_dim0_list:
        bx = lx//d0 if lx % d0 == 0  else lx//d0+1
        tiling_spaces.append([d0,bx,1,1,tx,1,1])


  elif (len(in_shape) - 1) in reduce_axis:
    """reduce-x"""
    possible_tx_list = [2**i for i in range(4,11)]
    for tx in possible_tx_list:
      if tx > lx: break
      ty = 1
      by = ly
      possible_dim1_list = [d1 for d1 in range(tx, lx+1, tx)]
      if possible_dim1_list[-1] != lx: possible_dim1_list.append(lx)
      for d1 in possible_dim1_list:
        bx = lx//d1 if lx % d1 == 0 else lx//d1+1
        tiling_spaces.append([1,d1,bx,by,1,tx,ty,1])

  else:
    """reduce-y"""
    tx = min(32,lx)
    bx = lx//tx if lx %tx==0 else lx//tx + 1
    d0 = tx
    for ty in range(min(8,ly),1025):
      if ty * tx > 1024: break
      possible_dim1_list = [d1 for d1 in range(ty, ly+1, ty)]
      for d1 in possible_dim1_list:
        by = ly//d1 if ly % d1 == 0 else ly//d1+1
        tiling_spaces.append([d0,d1,bx,by,1,tx,ty,1])

  input_type = namedtuple(op_type, dim_names)
  space = ListConfigSpace(input_type)
  if len(tuning_attrs_info[0]) != 0:
    for tiling_space in tiling_spaces:
      for tuning_attrs_config in tuning_attrs_info[1]:
        tmp = tiling_space[:]
        tmp.extend(tuning_attrs_config)
        config = input_type(*tmp)
        space.add(config)
  else:
      for tiling_space in tiling_spaces:
          config = input_type(*tiling_space)
          space.add(config)
  return space_res.index_table, space, key, expect, input_for_mod