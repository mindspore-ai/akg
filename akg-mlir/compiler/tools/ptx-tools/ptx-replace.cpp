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
   * 	// ld.global.nc.f32 	%f1, [%rd15+12];
   * 	// ld.global.nc.f32 	%f2, [%rd15+8];
   * 	// ld.global.nc.f32 	%f3, [%rd15+4];
   * 	// ld.global.nc.f32 	%f4, [%rd15];
   *  --------------- New -----------------
   * 	ld.global.nc.v4.f32     {%f4, %f3, %f2, %f1}, [%rd15];
   *
   *  ------------ Original ---------------
   * // st.global.f32 	[%rd18+12], %f12;
   * // st.global.f32 	[%rd18+8], %f11;
   * // st.global.f32 	[%rd18+4], %f10;
   * // st.global.f32 	[%rd18], %f9;
   *  --------------- New -----------------
   * st.global.v4.f32        [%rd18], {%f9, %f10, %f11, %f12};
   *
   * @param LdStGlobalCache pack of original load/store instructions
   * @return std::string new instruction that use vector load/store
   */
  std::string tryEmitVectorize(std::deque<std::string> LdStGlobalCache) const {
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
    for (auto ld : LdStGlobalCache) {
      std::vector<std::string> result = splitEachLoadStore(ld);
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
  std::vector<std::string> splitEachLoadStore(const std::string &line) const {
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
        for (auto subToken : subResult) {
          finalResult.push_back(subToken);
        }
      } else {
        finalResult.push_back(token);
      }
    }
    return finalResult;
  }

  std::pair<std::string, int> emitPackDataStr(std::map<int, std::string> srcIndex) const {
    std::string delimiter = ", ";
    std::string packDataStr = "{";
    int currSize = -1;
    int firstOffset = -1;
    for (auto it : srcIndex) {
      if (firstOffset == -1) {
        firstOffset = it.first;
      }
      if (currSize != -1 && currSize + static_cast<int>(vectorizeSize) != it.first) {
        std::cout << "Error, vectorize offset should be 4, get " << currSize << " vs " << it.first << "\n";
        (void)std::make_pair("", -1);
      }
      currSize = it.first;
      packDataStr += (it.second + delimiter);
    }
    // remove last ", "
    (void)packDataStr.erase(packDataStr.end() - delimiter.size());
    packDataStr += "}";
    return std::make_pair(packDataStr, firstOffset);
  }

  std::string emitDestStr(const std::string &dest, int firstOffset) const {
    auto destStr = "[" + dest;
    if (firstOffset != 0) {
      destStr += "+" + std::to_string(firstOffset);
    }
    destStr += "]";
    return destStr;
  }

  std::string emitInstruction(const std::string instruction, bool isLoad) const {
    std::string newInstruction;
    if (ncFlag && isLoad) {
      newInstruction = replaceString(instruction, "global", "global.nc.v4");
    } else {
      newInstruction = replaceString(instruction, "global", "global.v4");
    }
    return newInstruction;
  }
};

std::vector<std::vector<int>> parse2DArrayFromFile(const std::string &filename) {
  std::ifstream infile(filename);
  std::string line;
  std::string fileContent;
  std::vector<std::vector<int>> result;

  // Read the entire file content
  while (std::getline(infile, line)) {
    fileContent += line;
  }

  std::istringstream iss(fileContent);
  std::string val;

  while (std::getline(iss, val, '[')) {
    if (std::getline(iss, val, ']')) {
      std::istringstream issRow(val);
      std::vector<int> row;
      std::string num;

      while (std::getline(issRow, num, ',')) {
        // Remove any leading '['
        if (num.front() == '[') {
          (void)num.erase(num.begin());
        }
        row.push_back(std::stoi(num));
      }

      result.push_back(row);
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
  } else {
    return "";
  }
}

// .param .u64 Fused_BroadcastTo_inplace_assign_builder_15920035459442552540_kernel_param_0,
std::string getParam(const std::string &line, const std::string &value) {
  std::string patternStr = "\\s*\\.param\\s+\\.u64\\s+" + value + "_param_(\\d+)\\s*";
  std::regex pattern(patternStr);
  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    return match[1].str();
  } else {
    return "";
  }
}

// ld.param.u64   %rd2, [Fused_Reshape_Cast_Neg_Mul_fusion_18315353371220478878_kernel_param_18];
std::tuple<std::string, std::string> getRegFromLoadParamGlobal(const std::string &line, const std::string &value) {
  std::string patternStr = "\\s*ld\\.param\\.u64\\s+(\\%\\w+), \\[" + value + "_param_(\\w+)\\]\\s*;";
  std::regex pattern(patternStr);
  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    return std::make_tuple(match[1].str(), match[2].str());
  } else {
    return std::make_tuple("", "");
  }
}

bool containsInstruction(const std::string &line, const std::string &instruction) {
  return line.find(instruction) != std::string::npos;
}

void paramsToValues(std::vector<std::vector<int>> shapeArgs, std::vector<int> &values) {
  values.clear();
  for (size_t i = 0; i < shapeArgs.size(); i++) {
    values.push_back(kShouldRemove);
    values.push_back(kShouldKeep);  // real pointer
    for (size_t j = 0; j < shapeArgs[i].size(); j++) {
      values.push_back(shapeArgs[i][j]);
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
  int pos = 0;
  for (; iter != end; ++iter) {
    auto s = iter->str();
    std::smatch match;
    if (std::regex_search(s, match, patternNum) && match.size() > 1) {
      std::string matchStr = match.str(1);
      auto num = std::stoi(matchStr);
      posFlags.push_back(num);
    } else {
      posFlags.push_back(-1);
    }
    splitedStrs.push_back(s);
    pos++;
  }
}

void concatPtx(std::string &result, const std::vector<std::string> &vec, const std::vector<int> &posFlags,
               const std::vector<std::string> &valueStrList) {
  for (size_t idx = 0; idx < vec.size(); idx++) {
    if (posFlags[idx] != -1) {
      (void)result.append(valueStrList[(size_t)posFlags[idx]]);
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

void ptxReplacement(const std::string &inputFilename, const std::string &shapeArgFilename,
                    const std::string &outputFilename, const bool &ncFlag, const bool &dynFlag) {
  auto shapeArgs = parse2DArrayFromFile(shapeArgFilename);

  std::ifstream inFile(inputFilename);
  if (!inFile) {
    std::cerr << "Failed to open " << inputFilename << " for reading.\n";
    return;
  }

  std::ofstream outFile(outputFilename);
  std::ostringstream oss;
  if (!outFile) {
    std::cerr << "Failed to open " << outputFilename << " for writing.\n";
    return;
  }

  std::string kernelName;

  std::vector<int> valueList;
  std::vector<std::string> valueStrList;
  size_t totalTensorNums = shapeArgs.size();
  paramsToValues(shapeArgs, valueList);
  for (auto item : valueList) {
    valueStrList.push_back(std::to_string(item));
  }

  size_t currentTensor = 0;

  std::string line;
  int step = 0;
  std::deque<std::string> LdStGlobalCache;

#ifdef ENABLE_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  while (std::getline(inFile, line)) {
    bool diffInstruction = false;
    if (containsInstruction(line, "ld.global") || containsInstruction(line, "st.global")) {
      // We put all sequential ld/st instructions into cache
      if (!LdStGlobalCache.empty()) {
        auto lastLine = LdStGlobalCache.back();
        auto conflict = (containsInstruction(lastLine, "ld.global") && containsInstruction(line, "st.global")) ||
                        (containsInstruction(lastLine, "st.global") && containsInstruction(line, "ld.global"));
        if (!conflict) {
          LdStGlobalCache.push_back(line);
          continue;
        } else {
          diffInstruction = true;
        }
      } else {
        LdStGlobalCache.push_back(line);
        continue;
      }
    } else {
      diffInstruction = true;
    }

    // And we start to process the cache when we met different instruction
    if (diffInstruction) {
      auto vecInst = VectorizeEmitter(ncFlag).tryEmitVectorize(LdStGlobalCache);
      if (!vecInst.empty()) {
        // We successfully replace the instructions in cache with one vectorized instruction
        // so we clean the cache to avoid generate them.
        oss << vecInst << '\n';
        LdStGlobalCache.clear();
      }
    }

    // The instructions in cache cannot be replaced by one vectorized instruction, so we restore them.
    while (!LdStGlobalCache.empty()) {
      auto ld = LdStGlobalCache.front();
      oss << ld << '\n';
      LdStGlobalCache.pop_front();
    }
    if (dynFlag) {
      oss << line << '\n';
      continue;
    }
    if (step == 0 && containsInstruction(line, ".entry")) {
      kernelName = getKernelName(line);
      step = 1;
    } else if (step == 1) {
      if (containsInstruction(line, ".param")) {
        auto numStr = getParam(line, kernelName);
        auto num = std::stoi(numStr);
        if (valueList[num] == kShouldKeep) {
          currentTensor++;
          if (currentTensor == totalTensorNums) {
            size_t pos = line.find(",");
            if (pos != std::string::npos) {
              (void)line.erase(pos, 1);
            }
          }
          oss << line << '\n';
        }
        continue;
      } else {
        step = 2;
      }
    } else if (step == 2) {
      if (containsInstruction(line, kernelName)) {
        std::string reg, numStr;
        std::tie(reg, numStr) = getRegFromLoadParamGlobal(line, kernelName);
        if (reg != "" && numStr != "") {
          int num = std::stoi(numStr);
          if (valueList[num] != kShouldKeep) {
            oss << "\tmov.u64 " << reg << ", "
                << "REPLACEMARK" << num << ";\n";
            continue;
          }
        }
      }
      if (ncFlag && line.find("ld.global") != std::string::npos) {
        oss << addNcMarkForLdg(line) << "\n";
        continue;
      }
    }
    oss << line << '\n';
  }
  std::vector<int> posFlags;
  std::vector<std::string> splitedStrs;
  std::string originalPtx = oss.str();
  strSplitAndMark(originalPtx, splitedStrs, posFlags);

#ifdef ENABLE_PROFILE
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by ptx-replace in compile: " << duration.count() << " microseconds" << std::endl;
  start = std::chrono::high_resolution_clock::now();
#endif

  std::string updatedPtx;
  const size_t remSize = 100;
  updatedPtx.reserve(originalPtx.length() + remSize);
  concatPtx(updatedPtx, splitedStrs, posFlags, valueStrList);
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
    std::cerr << "Usage: " << argv[0] << " <input file> <output file>\n";
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
