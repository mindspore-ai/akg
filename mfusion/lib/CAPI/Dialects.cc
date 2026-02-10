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

#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir-c/BuiltinAttributes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Mfuse, mfuse, mlir::mfuse::MfuseDialect)

bool mlirAttributeIsAMfuseDeviceAttr(MlirAttribute attr) { return llvm::isa<mlir::mfuse::DeviceAttr>(unwrap(attr)); }

MlirTypeID mlirMfuseDeviceAttrGetTypeID(void) { return wrap(mlir::mfuse::DeviceAttr::getTypeID()); }

MlirAttribute mlirMfuseDeviceAttrGetDeviceType(MlirAttribute attr) {
  auto deviceAttr = mlir::cast<mlir::mfuse::DeviceAttr>(unwrap(attr));
  mlir::StringAttr typeAttr = deviceAttr.getDeviceType();
  return wrap(mlir::cast<mlir::Attribute>(typeAttr));
}

MlirAttribute mlirMfuseDeviceAttrGetIndex(MlirAttribute attr) {
  return wrap(mlir::cast<mlir::mfuse::DeviceAttr>(unwrap(attr)).getIndex());
}
