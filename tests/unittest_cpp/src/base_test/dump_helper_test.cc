/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <gtest/gtest.h>
#include "base/dump_helper.h"

namespace akg {
TEST(UTRegxMatch, RegxMatchHex) {
  EXPECT_EQ(UTRegxMatch::RegxMatchHex("0x123"), true);
  EXPECT_EQ(UTRegxMatch::RegxMatchHex("0xABC"), true);
  EXPECT_EQ(UTRegxMatch::RegxMatchHex("0XABC"), true);
  EXPECT_EQ(UTRegxMatch::RegxMatchHex("0x"), false);
}

TEST(UTDumpHelper, RegxMatchPlaceholder) {
  EXPECT_EQ(UTDumpHelper::RegxMatchPlaceholder("placeholder(input, 0x1234abcd)", "input"), true);
}
}  // namespace akg
