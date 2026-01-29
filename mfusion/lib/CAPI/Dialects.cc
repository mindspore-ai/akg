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

#include "mfusion/Dialect/Muse/MuseDialect.h"
#include "mfusion/Dialect/Muse/Muse.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir-c/BuiltinAttributes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Muse, muse, mlir::muse::MuseDialect)

bool mlirAttributeIsAMuseDeviceAttr(MlirAttribute attr) { return llvm::isa<mlir::muse::DeviceAttr>(unwrap(attr)); }

MlirTypeID mlirMuseDeviceAttrGetTypeID(void) { return wrap(mlir::muse::DeviceAttr::getTypeID()); }

MlirAttribute mlirMuseDeviceAttrGetDeviceType(MlirAttribute attr) {
  auto deviceAttr = mlir::cast<mlir::muse::DeviceAttr>(unwrap(attr));
  mlir::StringAttr typeAttr = deviceAttr.getDeviceType();
  return wrap(mlir::cast<mlir::Attribute>(typeAttr));
}

MlirAttribute mlirMuseDeviceAttrGetIndex(MlirAttribute attr) {
  return wrap(mlir::cast<mlir::muse::DeviceAttr>(unwrap(attr)).getIndex());
}
