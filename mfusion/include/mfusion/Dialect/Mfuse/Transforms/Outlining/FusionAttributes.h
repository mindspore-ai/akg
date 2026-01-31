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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_OUTLINING_FUSIONATTRIBUTES_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_OUTLINING_FUSIONATTRIBUTES_H

#include "llvm/ADT/StringRef.h"

/// Attribute names for fusion outlining and DVM conversion passes
/// All attributes use the "mfusion." prefix to avoid conflicts
namespace mfusion_attrs {

/// Marks a function as an outlined fusion region
/// Value type: UnitAttr (presence indicates outlined status)
static constexpr llvm::StringRef kOutlined = "mfusion.outlined";

/// Specifies the fusion type/kernel generator
/// Value type: StringAttr with enum values: "dvm", "akg"
static constexpr llvm::StringRef kFusionType = "mfusion.fusion_type";

/// Indicates whether a DVM subgraph uses dynamic shapes
/// Value type: BoolAttr
static constexpr llvm::StringRef kIsDynamic = "mfusion.is_dynamic";

/// Name of the copied subgraph function that remains in the IR after DVM lowering.
/// Value type: StringAttr
static constexpr llvm::StringRef kCopiedSubgraph = "mfusion.copied_subgraph";

} // namespace mfusion_attrs

#endif // MFUSION_DIALECT_MFUSE_TRANSFORMS_OUTLINING_FUSIONATTRIBUTES_H
