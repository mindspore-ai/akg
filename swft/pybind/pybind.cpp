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

PYBIND11_MODULE(c_expression, m) {
  py::class_<Value>(m, "Value");
  py::class_<Tensor, Value>(m, "Tensor")
    .def(py::init<const std::string &, const std::string &, const py::list &, const std::string &, const bool>())
    .def(py::init<py::array &, const std::string &, const bool>())
    .def("update_param", &Tensor::updateParam)
    .def("update_name", &Value::updateName)
    .def("update_position", &Value::updatePosition)
    .def("get_name", &Value::getName)
    .def("_get_shape",
         [](Tensor &self) {
           // Port std::vector<Scalar<int>> into std::vector<Scalar<double>>
           std::vector<Scalar<double>> shape_double;
           for (const auto &dim : self.getShape()) {
             if (dim.hasValue()) {
               shape_double.push_back(Scalar<double>("INT32", dim.getValue()));
             } else {
               Scalar<double> dim_double = Scalar<double>("INT32");
               if (dim.hasTile()) {
                 dim_double.setTile(dim.getTile());
               }
               dim_double.updateName(dim.getName());
               shape_double.push_back(dim_double);
             }
           }
           return shape_double;
         })
    .def("_get_dtype", &Tensor::getDTypePy)
    .def("_get_mem_type", &Tensor::getMemTypePy)
    .def("_get_format", &Tensor::getFormatPy)
    .def("_get_multi_core", &Tensor::isMultiCore)
    .def("sync_device_to_host", &Tensor::syncDeviceToHost)
    .def("sync_host_to_device", &Tensor::syncHostToDevice)
    .def("as_numpy", &Tensor::asNumpy)
    .def("host_data_ptr", &Tensor::getHostDataPtr, py::return_value_policy::reference)
    .def("device_data_ptr", &Tensor::getDeviceDataPtr, py::return_value_policy::reference)
    .def_property_readonly("type", &Tensor::getClassType);
  py::class_<Scalar<double>, Value>(m, "Scalar")
    .def(py::init<const std::string &>())
    .def(py::init<const std::string &, const float>())
    .def("update_name", &Value::updateName)
    .def("update_position", &Value::updatePosition)
    .def("get_name", &Value::getName)
    .def("has_value", &Scalar<double>::hasValue)
    .def("_has_tile", &Scalar<double>::hasTile)
    .def("_get_tile", &Scalar<double>::getTile)
    .def("_set_tile", &Scalar<double>::setTile)
    .def("_get_dtype", &Scalar<double>::getDtypeStr)
    .def("__deepcopy__", &Scalar<double>::deepcopy, py::arg("memo"))
    .def("__repr__", &Scalar<double>::toString)
    .def_property_readonly("type", &Scalar<double>::getClassType)
    .def_property_readonly("value", &Scalar<double>::getValue);
  py::class_<NPUSession>(m, "NPUSession")
    .def(py::init<const int>())
    .def("_get_stream", &NPUSession::getStream, py::return_value_policy::reference)
    .def("_get_current_device", &NPUSession::getCurrentDevice)
    .def("_sync_stream", &NPUSession::syncStream);
  m.def("new_subkernel", &newSubKernel);
  m.def("new_synckernel", &newSyncSubKernel);
  m.def("set_context", &setContext, py::arg("chip_type"), py::arg("code_type") = "ASCENDC");
  m.def("get_context", &getContext);
  m.def("push_to_list", &pushToList);
  m.def("compile_ckernel", &compileKernel);
  m.def("can_fit_memory", &canFitMemory);
}
