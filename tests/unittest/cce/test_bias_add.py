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

import json
from akg.ms import compilewithjson
from common import completeJson
from common import cce_run
from test_run.bias_add_run import bias_add_run


def configJson():
    dict_json = {"name": "BiasAdd"}
    inputs = []
    inputs.append({"name": "x", "shape": [1,1,1,1], "data_type": "float32"})
    inputs.append({"name": "b", "shape": [1], "data_type": "float32"})
    attr = {"name": "data_format", "value": ["NHWC", "NHWC"]}

    return json.dumps(completeJson(dict_json, inputs, [attr]))

if __name__ == "__main__":
    module = compilewithjson(configJson())
    cce_run(bias_add_run, [1, 1, 1, 1], "NHWC", "float32", {'mod': module})

