# Copyright 2019 Huawei Technologies Co., Ltd
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

import os
import logging
from tests.common.gen_random import random_gaussian
import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op import vector_matmul


logging.basicConfig(level=logging.DEBUG)


def np_matmul(matrix_a, matrix_b, trans_a=False, trans_b=False):
    if trans_a:
        matrix_a = matrix_a.transpose(1, 0)
    if trans_b:
        matrix_b = matrix_b.transpose(1, 0)
    m, k_a = matrix_a.shape
    k_b, n = matrix_b.shape
    if k_a != k_b:
        raise RuntimeError("matrix_a: %d %d vs matrix_b: %d %d" % (m, k_a, k_b, n))
    result = np.dot(matrix_a, matrix_b)
    return result


def gen_data(m, n, k, trans_a, trans_b, dtype):
    shape_x, shape_y = vector_matmul.get_shape(m, n, k, trans_a, trans_b)
    matrix_a = random_gaussian(shape_x, miu=0.5, sigma=0.01).astype(dtype)
    # matrix_b = random_gaussian(shape_y, miu=0.5, sigma=0.01).astype(dtype)
    # matrix_a = np.ones(shape_x, dtype=dtype)
    matrix_b = np.ones(shape_y, dtype=dtype)
    res = np_matmul(matrix_a, matrix_b, trans_a, trans_b)
    return matrix_a, matrix_b, res


def get_name(caseIndex=1, name="leftMatrix", m=0, n=0, k=0, trans_a=False, trans_b=False):
    res = "{}_{}_{}_{}_{}_{}_{}.bin".format(caseIndex, name, m, n, k, trans_a, trans_b)
    return res


def read_from_file(case_index, m, n, k, trans_a, trans_b, dtype):
    cur_path = os.path.abspath('.')
    benchmark_path, tmp = cur_path.split("ci_test")
    benchmark_path += "ci_test/AT-Benchmark/poly_benchmark/vector_matmul_benchmark/"

    # print benchmark_path
    left_matrix_name = get_name(case_index, "leftMatrix", m, n, k, trans_a, trans_b)
    right_matrix_name = get_name(case_index, "rightMatrix", m, n, k, trans_a, trans_b)
    result_name = get_name(case_index, "result", m, n, k, trans_a, trans_b)

    m_a_shape, m_b_shape = vector_matmul.get_shape(m, n, k, trans_a, trans_b)
    m_a = np.fromfile(benchmark_path + left_matrix_name, dtype=dtype).reshape(m_a_shape)
    m_b = np.fromfile(benchmark_path + right_matrix_name, dtype=dtype).reshape(m_b_shape)
    res_shape = (m, n)
    res = np.fromfile(benchmark_path + result_name, dtype=dtype).reshape(res_shape)
    return m_a, m_b, res


def vector_matmul_data(case_index, m, n, k, trans_a, trans_b, read_data, dump_data, dtype, debug_logging=False):
    m_a = ()
    m_b = ()
    bench_mark = ()

    if read_data:
        logging.debug("read from file!")
        m_a, m_b, bench_mark = read_from_file(case_index, m, n, k, trans_a, trans_b, dtype)
    else:
        m_a, m_b, bench_mark = gen_data(m, n, k, trans_a, trans_b, dtype)

    if dump_data:
        left_matrix_name = get_name(case_index, "leftMatrix", m, n, k, trans_a, trans_b)
        right_matrix_name = get_name(case_index, "rightMatrix", m, n, k, trans_a, trans_b)
        result_name = get_name(case_index, "result", m, n, k, trans_a, trans_b)
        m_a.tofile(left_matrix_name)
        m_b.tofile(right_matrix_name)
        bench_mark.tofile(result_name)

    if debug_logging:
        logging.debug("m_a shape:{}".format(m_a.shape))
        logging.debug("m_b shape:{}".format(m_b.shape))
        logging.debug(type(m_a))

    return m_a, m_b, bench_mark


def result_compare(actual, bench_mark, batch_tuple, M, N, K, r_tol=5e-3):
    output_shape = (M, N)

    error = 0
    count = 0
    lastErr = -2
    continueErr = 0
    maxContinue = -1
    maxEnd = 0
    logging.debug(actual.shape)
    logging.debug(bench_mark.shape)

    for m in range(output_shape[0]):
        for n in range(output_shape[1]):
            a = actual[m, n]
            b = bench_mark[m, n]
            if(abs(a - b) > abs(b) * r_tol):
                error += 1

                if lastErr + 1 == count:
                    continueErr += 1
                else:
                    if continueErr > maxContinue:
                        maxContinue = continueErr
                        maxEnd = lastErr
                    continueErr = 1
                lastErr = count

                # if a != 0.0:
                logging.debug("count: %6d expect: %20f actual: %20f %20.2f%%" % (count, b, a, abs(b - a) / b * 100))
            count += 1
    if continueErr > maxContinue:
        maxContinue = continueErr
        maxEnd = lastErr
    logging.debug("error num: %d/%d (%.2f%%)" % (error, count, 100.0 * error / count))
    logging.debug("longest error range: [%d, %d]" % (maxEnd - maxContinue + 1, maxEnd))
    if maxContinue >= 16:
        return False
    logging.debug("\n\n******************** test ok *****************\n\n")
    return True


def vector_matmul_run(case_index, m, n, k, trans_a, trans_b, read_data, dump_data, dtype, kernel_name, attrs):
    batch_tuple = (1, )
    # m = (m+15)//16*16
    # n = (n+15)//16*16
    # k = (k+15)//16*16

    mod, out_shape = vector_matmul.vector_matmul(m, n, k, trans_a, trans_b, dtype, kernel_name, attrs)
    utils.create_code(kernel_name, "./", mod.imported_modules[0].get_source())

    # Generate data
    m_a, m_b, bench_mark = vector_matmul_data(case_index, m, n, k, trans_a, trans_b, read_data, dump_data, dtype)

    # mod launch
    output = np.full(out_shape, np.nan, dtype=dtype)
    output = utils.mod_launch(mod, (m_a, m_b, output), expect=batch_tuple)

    # compare result
    compare_result = result_compare(output, bench_mark, batch_tuple, m, n, k, r_tol=1e-2)
    return (m_a, m_b), output, bench_mark, compare_result
