/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "pybind.h"
namespace py = pybind11;

PYBIND11_MODULE(c_expression, m)
{
    py::class_<Value>(m, "Value");
    py::class_<Tensor, Value>(m, "Tensor")
        .def(py::init<const std::string &, const std::string &, const py::list &, const std::string &, const bool>())
        .def("update_param", &Tensor::updateParam)
        .def("update_name", &Value::updateName)
        .def("get_name", &Value::getName)
        .def("update_position", &Value::updatePosition)
        .def("getShape", &Tensor::getShapePy)
        .def("getDtype", &Tensor::getDTypePy)
        .def("getMemType", &Tensor::getMemTypePy)
        .def("getFormat", &Tensor::getFormatPy)
        .def("getMultiCore", &Tensor::isMultiCore);
    py::class_<Scalar, Value>(m, "Scalar")
        .def(py::init<const std::string &>())
        .def(py::init<const std::string &, const float>())
        .def("update_name", &Value::updateName)
        .def("get_name", &Value::getName)
        .def("update_position", &Value::updatePosition)
        .def("has_value", &Scalar::hasValue)
        .def("getValue", &Scalar::getValue)
        .def("getDtype", &Scalar::getDtypeStr);
    m.def("new_subkernel", &newSubKernel);
    m.def("new_synckernel", &newSyncSubKernel);
    m.def("set_context", &setContext);
    m.def("get_context", &getContext);
    m.def("push_to_list", &pushToList);
    m.def("compile_ckernel", &compileKernel);
    m.def("can_fit_memory", &canFitMemory);
}
