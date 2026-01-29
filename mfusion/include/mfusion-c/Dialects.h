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

#ifndef MFUSION_C_DIALECTS_H
#define MFUSION_C_DIALECTS_H

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Muse, muse);

// Muse DeviceAttr CAPI
MLIR_CAPI_EXPORTED bool mlirAttributeIsAMuseDeviceAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID mlirMuseDeviceAttrGetTypeID(void);
MLIR_CAPI_EXPORTED MlirAttribute mlirMuseDeviceAttrGetDeviceType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirMuseDeviceAttrGetIndex(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // MFUSION_C_DIALECTS_H
