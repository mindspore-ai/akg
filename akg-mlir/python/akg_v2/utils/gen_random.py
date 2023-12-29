# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""generate gaussian random array"""

import os
import random
import logging
import sys
import time
import functools
from multiprocessing import Pool
from itertools import repeat
import numpy as np

RANDOM_SEED_NUM = 20


def func_time_required(func_name):
    """Checking the Time Required for Function Running."""

    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func_name(*args, **kwargs)
        t1 = time.time()
        logging.info("func_time_required func:%s, running:%lf seconds",
                     func_name.__name__, (t1 - t0))
        return result

    return wrapper


def get_profiling_mode():
    """get profiling mode."""
    env_dic = os.environ
    if env_dic.get('PROFILING_MODE') and env_dic.get('PROFILING_MODE').lower() == "true":
        return True
    return False


def func(size_, miu_=0, sigma_=8, seed_=None):
    """
    Select random func according to RANDOM_FUNC_MODE and randint, calculated by the length of the random_func_list.

    Args:
        size_ (int): Size of data.
        miu_ (int): Mean value. Default: 0.
        sigma_ (int): Standard deviation. Default: 8.
        seed_ (int): seed for random.

    Returns:
        Random func, from random_func_list.
    """
    size_ = (size_ + RANDOM_SEED_NUM - 1) // RANDOM_SEED_NUM
    random_func_list = [
        np.random.RandomState(seed_).normal(miu_, sigma_, size_),
        np.random.RandomState(seed_).logistic(miu_, sigma_, size_),
        np.random.RandomState(seed_).laplace(miu_, sigma_, size_),
        np.random.RandomState(seed_).uniform(miu_, sigma_, size_),
        np.random.RandomState(seed_).tomaxint(size_),
    ]
    env_dic = os.environ
    if not env_dic.get('RANDOM_FUNC_MODE'):
        func_idx = 0
    else:
        func_idx = np.random.RandomState(None).randint(len(random_func_list))
    res = random_func_list[func_idx]
    return res


@func_time_required
def random_gaussian(size, miu=0, sigma=8, epsilon=0, seed=None):
    """Generate random array with absolution value obeys gaussian distribution."""
    random_data_disk_path = None
    if os.environ.get("RANDOM_DATA_DISK_PATH") is not None:
        random_data_disk_path = os.environ.get("RANDOM_DATA_DISK_PATH") + "/random_data_%s_%s.bin" % (
            str(miu), str(sigma))

    if random_data_disk_path is None or (not os.path.exists(random_data_disk_path)):
        if sigma <= 0:
            sys.stderr.write(
                "Error: Expect positive sigmal for gaussian distribution. but get %f\n" % sigma)
            sys.exit(1)
        size_c = 1
        for x in size:
            size_c = size_c * x

        if seed is None:
            seed_ = []
            for i in range(RANDOM_SEED_NUM):
                now = int(time.time() % 10000 * 10000) + random.randint(i, 100)
                seed_.append(now)
        else:
            seed_ = [seed] * RANDOM_SEED_NUM
        logging.debug("random_gaussian seeds: %d", seed_)
        # In the profiling scenario, when a new process is used to run test cases, data generated by multiple processes
        # stops responding. To locate the fault, please set this condition to False.
        if not bool(get_profiling_mode()):
            with Pool(processes=8) as pool:
                ret = np.array(pool.starmap(
                    func, zip(repeat(size_c), repeat(miu), repeat(sigma), seed_)))
        else:
            numbers = list()
            for s in seed_:
                numbers.extend(func(size_c, miu, sigma, s))
            ret = np.array(numbers)
        ret = ret.flatten()
        return ret[:size_c].reshape(size) + epsilon

    data_len = functools.reduce(lambda x, y: x * y, size)
    data_pool = np.fromfile(random_data_disk_path)
    if data_len % len(data_pool) != 0:
        copy_num = (data_len // len(data_pool)) + 1
    else:
        copy_num = data_len // len(data_pool)
    data_copy = np.copy(data_pool)
    data_copy_list = []
    for _ in range(copy_num):
        np.random.shuffle(data_copy)
        data_copy_list.append(data_copy)
    data_pool = np.concatenate(tuple(data_copy_list), axis=0)
    return data_pool[0:data_len].reshape(size) + epsilon


def gen_epsilon(dtype):
    """Generate suggested epsilon according to data type."""
    return 1e-7 if dtype == np.float32 else 1e-3


def gen_indices_tensor_scatter_add(shape1, shape2, dtype2):
    """Generate indices for tensor_scatter_add."""
    if dtype2 != "int32":
        raise ValueError("Currently only support int32 indices")

    indices = np.zeros(shape2, dtype2)
    indices = indices.reshape(-1, indices.shape[-1])
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            # add outbounds situation
            if np.random.random() < 0.1:
                # less than 0 and larger than original shape
                indices[i][j] = - \
                    1 if np.random.random() < 0.5 else shape1[j] + 10
            else:
                indices[i][j] = np.random.randint(shape1[j], size=())
    indices = indices.reshape(shape2)
    return indices


def gen_indices_gather(shape1, shape2, dtype2, axis):
    """Generate indices for gather."""
    if dtype2 != "int32":
        raise ValueError("Currently only support int32 indices")
    indices = np.random.randint(
        low=0, high=shape1[axis], size=shape2).astype(dtype2)
    return indices


def gen_indices_unsorted_segment_sum(shape2, dtype2, num):
    """Generate indices for gather unsorted_segment_sum."""
    # currently only support 1D
    if dtype2 != "int32":
        raise ValueError("Currently only support int32 indices")
    return np.random.randint(low=0, high=num, size=shape2).astype(dtype2)


def gen_indices_gather_nd(shape1, shape2, dtype2):
    """Generate indices for gather nd."""
    out_dim1 = 1
    for i in range(len(shape2) - 1):
        out_dim1 = out_dim1 * shape2[i]
    if dtype2 != "int32":
        raise ValueError("Currently only support int32 indices")
    indices = np.zeros([shape2[-1], out_dim1]).astype(dtype2)
    for i in range(shape2[-1]):
        # add outbounds situation
        if np.random.random() < 0.1:
            if np.random.random() < 0.5:
                # less than 0
                indices[i] = np.random.randint(
                    low=0, high=shape1[i], size=out_dim1) - 10
            else:
                # larger than original shape
                indices[i] = np.random.randint(
                    low=0, high=shape1[i], size=out_dim1) + 10
        else:
            indices[i] = np.random.randint(
                low=0, high=shape1[i], size=out_dim1)

    indices = indices.transpose()
    indices = indices.reshape(shape2)
    return indices


def gen_indices(indices_argument):
    """Generate indices."""
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
        return gen_indices_unsorted_segment_sum(indices_shape, indices_dtype, attrs)
    if op_name != "TensorScatterAdd":
        raise ValueError("Input OP Name: {} Not Known!".format(op_name))
    return gen_indices_tensor_scatter_add(data_shape, indices_shape, indices_dtype)


def gen_csr_indices(indices_argument):
    """Generate csr indices."""
    data_shape = indices_argument.data_shape
    indices_shape = indices_argument.indices_shape
    indices_dtype = indices_argument.indices_dtype
    indptr_choice = np.arange(0, indices_shape[0], dtype=indices_dtype)
    indptr = np.sort(np.random.choice(
        indptr_choice, data_shape[0] - 1, replace=True))
    indptr = np.concatenate(
        (np.array([0], dtype=indices_dtype), indptr, np.array([indices_shape[0]], dtype=indices_dtype)))
    indices_choice = np.arange(data_shape[1], dtype=indices_dtype)
    idx_range = indices_choice.copy()
    idx_bound = np.diff(indptr).reshape(-1, 1)
    max_count = idx_bound.max().tolist()
    np.random.shuffle(indices_choice)
    indices_choice = sorted(indices_choice[: max_count])
    indices = np.where(idx_range[: max_count] <
                       idx_bound, indices_choice, data_shape[1])
    mask = np.less(indices, data_shape[1]).nonzero()
    indices = indices[mask]
    return indptr, indices
