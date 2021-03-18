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

"""operator dsl function: erf_ad"""

import akg
from tests.common.test_op import erf
from akg.utils import custom_tiling as ct_util, kernel_exec as utils


erf_ad_set_dim_map = {
    str(((1, 128), "float16", "cce_erf_fp16")): ((128, 128), (128, 128)),
    str(((128, 128), "float16", "cce_erf_fp16")): ((0, 0), (128, 128)),
    str(((128, 256), "float16", "cce_erf_fp16")): ((0, 0), (128, 128)),
}


def erf_ad_set_dim_func(head, x):
    key = []
    key.append(tuple(x.shape))
    key.append(x.dtype)
    hash_key = str(tuple(key))

    if hash_key in erf_ad_set_dim_map.keys():
        return ct_util.set_dims(erf_ad_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key


@ct_util.reg_set_dim_func(erf_ad_set_dim_func)
def erf_ad(head, x):
    """Compute gradient of erf operator using automatic differentiate."""
    if utils.product_is_mini():
        raise RuntimeError("Not support erf_ad on mini device.")
    output = erf.erf(x)
    jacs = list(akg.differentiate(output, [x], head))
    return jacs[0]
