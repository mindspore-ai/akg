/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Analysis/TypeUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
// append namedAttribute to origin type's encoding
RankedTensorType updateTensorEncodingAttr(RankedTensorType origin, NamedAttribute &attr) {
  if (!origin) {
    return origin;
  }
  DictionaryAttr dict = origin.getEncoding().dyn_cast_or_null<DictionaryAttr>();
  if (!dict) {
    // If the original encoding is empty or non-DictionaryAttr, initialize it.
    llvm::SmallVector<NamedAttribute> namedAttrs{attr};
    dict = DictionaryAttr::get(attr.getValue().getContext(), namedAttrs);
  } else {
    // append namedAttribute to origin type's encoding
    NamedAttrList attrList(dict);
    // if origin type already contains attr, erase it. OtherWise, nothing will
    // be done.
    (void)attrList.erase(attr.getName());
    attrList.append(attr);
    dict = attrList.getDictionary(attr.getValue().getContext());
  }
  return RankedTensorType::get(origin.getShape(), origin.getElementType(), dict);
}

}  // namespace mlir
