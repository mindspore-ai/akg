#pragma once

#include <cstdint>

constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct AddTilingData {
  uint32_t blockNum;
  uint64_t totalLength;
  uint64_t numPerCore;
  uint64_t tailNumLastCore;
};
