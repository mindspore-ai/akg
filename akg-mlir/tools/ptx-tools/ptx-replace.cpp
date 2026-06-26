/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace {
constexpr auto kShouldRemove = -9999999;
constexpr auto kShouldKeep = -10000000;

std::vector<std::string> splitString(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);

  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

std::string replaceString(const std::string &str, const std::string &oldSubstr, const std::string &newSubstr) {
  std::string result = str;
  size_t pos = 0;

  while ((pos = result.find(oldSubstr, pos)) != std::string::npos) {
    (void)result.replace(pos, oldSubstr.length(), newSubstr);
    pos += newSubstr.length();
  }

  return result;
}

struct VectorizeEmitter {
  explicit VectorizeEmitter(bool ncFlag) : ncFlag(ncFlag) {}
  bool ncFlag{false};

  /**
   * @brief
   * This Function will try to replace a pack of 4 ld/st instructions into one vectorized instruction
   * e.g.
   *  ------------ Original ---------------
   * // ld.global.nc.f32 %f1, [%rd15+12];
   * // ld.global.nc.f32 %f2, [%rd15+8];
   * // ld.global.nc.f32 %f3, [%rd15+4];
   * // ld.global.nc.f32 %f4, [%rd15];
   *  --------------- New -----------------
   * ld.global.nc.v4.f32 {%f4, %f3, %f2, %f1}, [%rd15];
   *
   *  ------------ Original ---------------
   * // st.global.f32 [%rd18+12], %f12;
   * // st.global.f32 [%rd18+8], %f11;
   * // st.global.f32 [%rd18+4], %f10;
   * // st.global.f32 [%rd18], %f9;
   *  --------------- New -----------------
   * st.global.v4.f32 [%rd18], {%f9, %f10, %f11, %f12};
   *
   * @param LdStGlobalCache pack of original load/store instructions
   * @return std::string new instruction that use vector load/store
   */
  [[nodiscard]] std::string tryEmitVectorize(const std::deque<std::string> &LdStGlobalCache) const {
    if (LdStGlobalCache.size() != vectorizeSize) {
      return "";
    }
    // We start to analyze each load/store instruction to get the final
    // vectorized instruction. During this process, we may exit due to
    // unsupported case.
    std::string instruction;
    std::string dest;
    // Note that we use map to sort the load/store src in a ascending
    std::map<int, std::string> srcIndex;
    auto isLoad = LdStGlobalCache.front().find("ld.global") != std::string::npos;
    auto instPos = 0;
    auto destPos = isLoad ? 2 : 1;
    auto srcPos = isLoad ? 1 : -1;
    for (auto it = LdStGlobalCache.cbegin(); it != LdStGlobalCache.cend(); ++it) {
      std::vector<std::string> result = splitEachLoadStore(*it);
      if (result.size() != maxSplitLen - 1 && result.size() != maxSplitLen) {
        return "";
      }
      if (instruction.empty()) {
        instruction = result[instPos];
      }
      if (instruction != result[instPos]) {
        // mismatch instruction
        return "";
      }
      if (dest.empty()) {
        dest = result[destPos];
      }
      if (dest != result[destPos]) {
        // mismatch dest
        return "";
      }

      int offset = -1;
      if (result.size() == 3) {
        offset = 0;
      } else if (result.size() == 4) {
        try {
          offset = std::stoi(result[destPos + 1]);
        } catch (const std::exception &e) {
          // offset is not a number
          std::cerr << "convert to int error, numStr is " << result[destPos + 1] << std::endl;
          return "";
        }
      }
      if (srcPos == -1) {
        srcIndex[offset] = result.back();
      } else {
        srcIndex[offset] = result[srcPos];
      }
    }

    // Now we can emit the final result:
    instruction = emitInstruction(instruction, isLoad);
    auto [packDataStr, firstOffset] = emitPackDataStr(srcIndex);
    if (packDataStr.empty()) {
      return "";
    }
    auto destStr = emitDestStr(dest, firstOffset);
    std::string vectorLoad;
    vectorLoad += "\t" + instruction + "\t";
    if (isLoad) {
      vectorLoad += (packDataStr + ", " + destStr);
    } else {
      vectorLoad += (destStr + ", " + packDataStr);
    }
    vectorLoad += ";\n";
    return vectorLoad;
  }

 private:
  // currently we only support vectorize length = 4;
  size_t vectorizeSize = 4;
  size_t maxSplitLen = 4;  // "instruction, dest, (offset), src"
  [[nodiscard]] std::vector<std::string> splitEachLoadStore(const std::string &line) const {
    std::vector<std::string> result = splitString(line, ' ');
    // we further remove all irrelevant symbols to get pure "instruction, dest, (offset), src"
    std::vector<std::string> finalResult;
    for (auto token : result) {
      token = replaceString(token, "\t", "");
      token = replaceString(token, " ", "");
      token = replaceString(token, ",", "");
      token = replaceString(token, ";", "");
      if (token.find('[') != std::string::npos) {
        // that is [dest + offset]
        token = replaceString(token, "[", "");
        token = replaceString(token, "]", "");
        std::vector<std::string> subResult = splitString(token, '+');
        std::copy(subResult.begin(), subResult.end(), std::back_inserter(finalResult));
      } else {
        finalResult.push_back(token);
      }
    }
    return finalResult;
  }

  [[nodiscard]] std::pair<std::string, int> emitPackDataStr(const std::map<int, std::string> &srcIndex) const {
    std::string delimiter = ", ";
    std::string packDataStr = "{";
    int currSize = -1;
    int firstOffset = -1;
    for (auto it = srcIndex.cbegin(); it != srcIndex.cend(); ++it) {
      if (firstOffset == -1) {
        firstOffset = it->first;
      }
      if (currSize != -1 && currSize + static_cast<int>(vectorizeSize) != it->first) {
        std::cout << "Error, vectorize offset should be 4, get " << currSize << " vs " << it->first << "\n";
        return std::make_pair("", -1);
      }
      currSize = it->first;
      packDataStr += (it->second + delimiter);
    }
    // remove last ", "
    (void)packDataStr.erase(packDataStr.end() - delimiter.size());
    packDataStr += "}";
    return std::make_pair(packDataStr, firstOffset);
  }

  [[nodiscard]] std::string emitDestStr(const std::string &dest, int firstOffset) const {
    auto destStr = "[" + dest;
    if (firstOffset != 0) {
      destStr += "+" + std::to_string(firstOffset);
    }
    destStr += "]";
    return destStr;
  }

  [[nodiscard]] std::string emitInstruction(const std::string &instruction, bool isLoad) const {
    std::string newInstruction;
    if (ncFlag && isLoad) {
      newInstruction = replaceString(instruction, "global", "global.nc.v4");
    } else {
      newInstruction = replaceString(instruction, "global", "global.v4");
    }
    return newInstruction;
  }
};

std::vector<int> parseRow(const std::string &rowStr) {
  std::istringstream issRow(rowStr);
  std::vector<int> row;
  std::string num;

  while (std::getline(issRow, num, ',')) {
    if (!num.empty() && num.front() == '[') {
      (void)num.erase(num.begin());
    }
    try {
      row.push_back(std::stoi(num));
    } catch (...) {
      std::cerr << "convert to int error, numStr is " << num << std::endl;
    }
  }

  return row;
}

std::vector<std::vector<int>> parse2DArrayFromFile(const std::string &filename) {
  std::ifstream infile(filename);
  std::string line;
  std::string fileContent;
  std::vector<std::vector<int>> result;

  while (std::getline(infile, line)) {
    fileContent += line;
  }

  std::istringstream iss(fileContent);
  std::string val;

  while (std::getline(iss, val, '[')) {
    if (std::getline(iss, val, ']')) {
      std::vector<int> row = parseRow(val);
      if (!row.empty()) {
        result.push_back(row);
      }
    }
  }

  return result;
}

// .visible .entry Fused_Reshape_Cast_Neg_Mul_fusion_18315353371220478878_kernel(
std::string getKernelName(const std::string &line) {
  std::string patternStr = "\\s*\\.visible\\s+\\.entry\\s+(\\w+)\\(";
  std::regex pattern(patternStr);
  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    return match[1].str();
  }
  return "";
}

// .param .u64 Fused_BroadcastTo_inplace_assign_builder_15920035459442552540_kernel_param_0,
std::string getParam(const std::string &line, const std::string &value) {
  std::string patternStr = "\\s*\\.param\\s+\\.u64\\s+" + value + "_param_(\\d+)\\s*";
  std::regex pattern(patternStr);
  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    return match[1].str();
  }
  return "";
}

// ld.param.u64   %rd2, [Fused_Reshape_Cast_Neg_Mul_fusion_18315353371220478878_kernel_param_18];
std::tuple<std::string, std::string> getRegFromLoadParamGlobal(const std::string &line, const std::string &value) {
  std::string patternStr = "\\s*ld\\.param\\.u64\\s+(\\%\\w+), \\[" + value + "_param_(\\w+)\\]\\s*;";
  std::regex pattern(patternStr);
  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    return std::make_tuple(match[1].str(), match[2].str());
  }
  return std::make_tuple("", "");
}

bool containsInstruction(const std::string &line, const std::string &instruction) {
  return line.find(instruction) != std::string::npos;
}

void paramsToValues(const std::vector<std::vector<int>> &shapeArgs, std::vector<int> &values) {
  values.clear();
  for (const auto &shapeArg : shapeArgs) {
    values.push_back(kShouldRemove);
    values.push_back(kShouldKeep);  // real pointer
    for (size_t j = 0; j < shapeArg.size(); j++) {
      values.push_back(shapeArg[j]);
    }
  }
}

std::string addNcMarkForLdg(std::string str) {
  if (str.find("ld.global.nc") != std::string::npos) {
    return str;
  }

  size_t pos = str.find("ld.global");
  if (pos != std::string::npos) {
    (void)str.replace(pos, 9, "ld.global.nc");
  }

  return str;
}

void strSplitAndMark(const std::string &input, std::vector<std::string> &splitedStrs, std::vector<int> &posFlags) {
  std::regex pattern("(REPLACEMARK\\d+)");
  std::sregex_token_iterator iter(input.begin(), input.end(), pattern, {-1, 1});
  std::sregex_token_iterator end;

  std::regex patternNum("REPLACEMARK(\\d+)");
  for (; iter != end; ++iter) {
    auto s = iter->str();
    std::smatch match;
    if (std::regex_search(s, match, patternNum) && match.size() > 1) {
      std::string matchStr = match.str(1);
      try {
        auto num = std::stoi(matchStr);
        posFlags.push_back(num);
      } catch (...) {
        posFlags.push_back(-1);
      }
    } else {
      posFlags.push_back(-1);
    }
    splitedStrs.push_back(s);
  }
}

void concatPtx(std::string &result, const std::vector<std::string> &vec, const std::vector<int> &posFlags,
               const std::vector<std::string> &valueStrList) {
  for (size_t idx = 0; idx < vec.size(); idx++) {
    if (posFlags[idx] != -1) {
      auto pos = static_cast<size_t>(posFlags[idx]);
      if (pos < valueStrList.size()) {
        (void)result.append(valueStrList[pos]);
      } else {
        // If index is out of bounds, keep the original REPLACEMARK string for debugging
        (void)result.append(vec[idx]);
        std::cerr << "Warning: posFlags[" << idx << "] = " << pos
                  << " exceeds valueStrList.size() = " << valueStrList.size() << "\n";
      }
    } else {
      (void)result.append(vec[idx]);
    }
  }
}

void copyFile(const std::string &inputFilename, const std::string &outputFilename) {
  std::ifstream inFile(inputFilename, std::ios::binary);
  std::ofstream outFile(outputFilename, std::ios::binary);
  outFile << inFile.rdbuf();
  inFile.close();
  outFile.close();
}

// Struct to hold processing state
struct ProcessingState {
  std::string kernelName;
  std::vector<int> valueList;
  std::vector<std::string> valueStrList;
  size_t totalTensorNums;
  size_t currentTensor;
  int step;
  bool ncFlag;
  bool dynFlag;
  std::ostringstream oss;
  std::deque<std::string> ldStCache;

  ProcessingState(std::vector<std::vector<int>> &shapeArgs, bool nc, bool dyn)
      : totalTensorNums(shapeArgs.size()), currentTensor(0), step(0), ncFlag(nc), dynFlag(dyn) {
    paramsToValues(shapeArgs, valueList);
    std::transform(valueList.begin(), valueList.end(), std::back_inserter(valueStrList),
                   [](int value) { return std::to_string(value); });
  }
};

// Process ld/st cache for vectorization
bool processLdStCache(ProcessingState &state, const std::string &line) {
  if (!state.dynFlag && (containsInstruction(line, "ld.global") || containsInstruction(line, "st.global"))) {
    bool addToCache = true;
    if (!state.ldStCache.empty()) {
      const std::string &lastLine = state.ldStCache.back();
      bool isMixed = (containsInstruction(lastLine, "ld.global") && containsInstruction(line, "st.global")) ||
                     (containsInstruction(lastLine, "st.global") && containsInstruction(line, "ld.global"));
      if (isMixed) {
        // Different type, flush cache first
        addToCache = false;
      }
    }

    if (addToCache) {
      state.ldStCache.push_back(line);
      return true;
    }
  }

  // Flush the cache
  std::string vecInst = VectorizeEmitter(state.ncFlag).tryEmitVectorize(state.ldStCache);
  if (!vecInst.empty()) {
    state.oss << vecInst;
  } else {
    for (auto it = state.ldStCache.cbegin(); it != state.ldStCache.cend(); ++it) {
      state.oss << *it << "\n";
    }
  }
  state.ldStCache.clear();
  return false;
}

// Handle step 0: find kernel name
void processStep0(ProcessingState &state, const std::string &line) {
  if (containsInstruction(line, ".entry")) {
    state.kernelName = getKernelName(line);
    state.step = 1;
  }
}

// Handle step 1: process .param lines
bool processStep1(ProcessingState &state, const std::string &line) {
  if (!containsInstruction(line, ".param")) {
    state.step = 2;
    return false;
  }

  std::string numStr = getParam(line, state.kernelName);
  if (numStr.empty()) {
    state.oss << line << "\n";
    return true;
  }

  try {
    int num = std::stoi(numStr);
    if (num < 0 || num >= static_cast<int>(state.valueList.size()) || state.valueList[num] != kShouldKeep) {
      return true;
    }
    state.currentTensor++;
    std::string processedLine = line;
    if (state.currentTensor == state.totalTensorNums) {
      size_t pos = processedLine.find(",");
      if (pos != std::string::npos) {
        processedLine.erase(pos, 1);
      }
    }
    state.oss << processedLine << "\n";
  } catch (const std::exception &) {
    state.oss << line << "\n";
  }
  return true;
}

bool tryReplaceParamLoad(ProcessingState &state, const std::string &reg, const std::string &numStr) {
  try {
    int num = std::stoi(numStr);
    if (num >= 0 && num < static_cast<int>(state.valueList.size()) && state.valueList[num] != kShouldKeep) {
      state.oss << "\tmov.u64 " << reg << ", REPLACEMARK" << num << ";\n";
      return true;
    }
  } catch (const std::exception &) {
    std::cerr << "convert to int error, numStr is " << numStr << std::endl;
  }
  return false;
}

// Handle step 2: process param loads and nc flag
bool processStep2(ProcessingState &state, const std::string &line) {
  if (containsInstruction(line, state.kernelName)) {
    std::string reg;
    std::string numStr;
    std::tie(reg, numStr) = getRegFromLoadParamGlobal(line, state.kernelName);
    if (!reg.empty() && !numStr.empty()) {
      if (tryReplaceParamLoad(state, reg, numStr)) {
        return true;
      }
    }
  }

  if (state.ncFlag && line.find("ld.global") != std::string::npos) {
    state.oss << addNcMarkForLdg(line) << "\n";
    return true;
  }
  return false;
}

// Process a single line of PTX
void processLine(ProcessingState &state, const std::string &line) {
  // First handle ld/st cache
  if (processLdStCache(state, line)) {
    return;
  }

  // If dynamic flag, just output the line
  if (state.dynFlag) {
    state.oss << line << "\n";
    return;
  }

  // Handle step processing
  bool handled = false;
  switch (state.step) {
    case 0:
      processStep0(state, line);
      break;
    case 1:
      handled = processStep1(state, line);
      break;
    case 2:
      handled = processStep2(state, line);
      break;
  }

  // If not handled, output as is
  if (!handled) {
    state.oss << line << "\n";
  }
}

void ptxReplacement(const std::string &inputFilename, const std::string &shapeArgFilename,
                    const std::string &outputFilename, const bool &ncFlag, const bool &dynFlag) {
  auto shapeArgs = parse2DArrayFromFile(shapeArgFilename);

  std::ifstream inFile(inputFilename);
  if (!inFile) {
    std::cerr << "Failed to open " << inputFilename << " for reading.\n";
    return;
  }

  std::ofstream outFile(outputFilename);
  if (!outFile) {
    std::cerr << "Failed to open " << outputFilename << " for writing.\n";
    return;
  }

  ProcessingState state(shapeArgs, ncFlag, dynFlag);

#ifdef ENABLE_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  // Process all lines
  std::string line;
  while (std::getline(inFile, line)) {
    processLine(state, line);
  }

  // Flush any remaining cache content
  if (!state.ldStCache.empty()) {
    std::string vecInst = VectorizeEmitter(state.ncFlag).tryEmitVectorize(state.ldStCache);
    if (!vecInst.empty()) {
      state.oss << vecInst;
    } else {
      for (auto it = state.ldStCache.cbegin(); it != state.ldStCache.cend(); ++it) {
        state.oss << *it << "\n";
      }
    }
  }

  // Split and replace
  std::vector<int> posFlags;
  std::vector<std::string> splitedStrs;
  std::string originalPtx = state.oss.str();
  strSplitAndMark(originalPtx, splitedStrs, posFlags);

#ifdef ENABLE_PROFILE
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by ptx-replace in compile: " << duration.count() << " microseconds" << std::endl;
  start = std::chrono::high_resolution_clock::now();
#endif

  // Concat and write
  std::string updatedPtx;
  const size_t remSize = 100;
  updatedPtx.reserve(originalPtx.length() + remSize);
  concatPtx(updatedPtx, splitedStrs, posFlags, state.valueStrList);

#ifdef ENABLE_PROFILE
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by ptx-replace in runtime: " << duration.count() << " microseconds" << std::endl;
#endif

  outFile << updatedPtx;
  inFile.close();
  outFile.close();
}
}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <input file> <shape_arg file> <output file>\n";
    return 1;
  }

  std::string inputFilename(argv[1]);
  std::string shapeArgFilename(argv[2]);
  std::string outputFilename(argv[3]);
  bool ncFlag = false;
  bool dynFlag = false;
  if (argc >= 5) {
    std::string nonCoherent(argv[4]);
    ncFlag = (nonCoherent == "nc");
  }

  if (argc >= 6) {
    std::string dynamicShape(argv[5]);
    dynFlag = (dynamicShape == "dynamic_shape");
  }

  std::ifstream fInput(inputFilename.c_str());
  if (!fInput.good()) {
    std::cout << "[ERROR] input file is not found, replacement exit. the path is '" << inputFilename << "'.\n";
    return 0;
  }

  std::ifstream fShape(shapeArgFilename.c_str());
  if (!fShape.good()) {
    copyFile(inputFilename, outputFilename);
    return 0;
  }

  ptxReplacement(inputFilename, shapeArgFilename, outputFilename, ncFlag, dynFlag);
  return 0;
}
