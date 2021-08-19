from .test_ms_tensor_scatter_add import tensor_scatter_add_np,\
     gen_indices_tensor_scatter_add
from .test_ms_gather import gather_np, gen_indices_gather
from .test_ms_unsorted_segment_sum import gen_indices_unsorted_segment_sum
from .test_ms_gather_nd import gen_indices_gather_nd

def gen_indices(indices_argument):
    op_name = indices_argument.name
    data_shape = indices_argument.data_shape
    indices_shape = indices_argument.indices_shape
    indices_dtype = indices_argument.indices_dtype
    attrs = indices_argument.attrs
    if op_name == "Gather":
        return gen_indices_gather(data_shape, indices_shape, indices_dtype, attrs)
    elif op_name == "GatherNd":
        return gen_indices_gather_nd(data_shape, indices_shape, indices_dtype)
    elif op_name == "UnsortedSegmentSum":
        return gen_indices_unsorted_segment_sum(data_shape, indices_shape, indices_dtype, attrs)
    assert op_name == "TensorScatterAdd", "Input OP Name Not Known!"
    return gen_indices_tensor_scatter_add(data_shape, indices_shape, indices_dtype)
