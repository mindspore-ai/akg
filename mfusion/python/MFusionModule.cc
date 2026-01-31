/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mfusion-c/Dialects.h"
#include "mfusion-c/Passes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir-c/BuiltinAttributes.h"

namespace py = pybind11;
using mlir::python::adaptors::mlir_attribute_subclass;

PYBIND11_MODULE(_mfusion, m) {
  mlirRegisterMFusionPasses();

  m.doc() = "mfusion python bindings";

  m.def(
    "register_mfuse_dialect",
    [](MlirContext context, bool load) {
      MlirDialectHandle handle = mlirGetDialectHandle__mfuse__();
      mlirDialectHandleRegisterDialect(handle, context);
      if (load) {
        mlirDialectHandleLoadDialect(handle, context);
      }
    },
    py::arg("context"), py::arg("load") = true, "Register MFUSE dialect");

  m.def(
    "register_dvm_dialect",
    [](MlirContext context, bool load) {
      MlirDialectHandle handle = mlirGetDialectHandle__dvm__();
      mlirDialectHandleRegisterDialect(handle, context);
      if (load) {
        mlirDialectHandleLoadDialect(handle, context);
      }
    },
    py::arg("context"), py::arg("load") = true, "Register DVM dialect");

  // Bind Mfuse::DeviceAttr through CAPI
  auto deviceAttrClass =
    mlir_attribute_subclass(m, "DeviceAttr", mlirAttributeIsAMfuseDeviceAttr, mlirMfuseDeviceAttrGetTypeID)
      .def_property_readonly(
        "device_type",
        [](MlirAttribute self) -> std::string {
          MlirAttribute typeAttr = mlirMfuseDeviceAttrGetDeviceType(self);
          MlirStringRef typeStr = mlirStringAttrGetValue(typeAttr);
          return std::string(typeStr.data, typeStr.length);
        },
        "Get the device type (e.g., 'cpu', 'npu')")
      .def_property_readonly(
        "index",
        [](MlirAttribute self) -> int64_t {
          MlirAttribute indexAttr = mlirMfuseDeviceAttrGetIndex(self);
          return mlirIntegerAttrGetValueInt(indexAttr);
        },
        "Get the device index");

}
