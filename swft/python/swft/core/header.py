#!/usr/bin/env python3
# coding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

data_utils = """#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <arm_neon.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include "acl/acl.h"


bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        printf("failed to get file, file path: %s", filePath.c_str());
        std::exit(0);
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        printf("%s is not a file, please enter a file", filePath.c_str());
        std::exit(0);
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        printf("Open file failed. path = %s", filePath.c_str());
        std::exit(0);
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        printf("file size is 0");
        file.close();
        std::exit(0);
        return false;
    }
    if (size > bufferSize) {
        printf("file size is larger than buffer size");
        file.close();
        std::exit(0);
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        printf("Write file failed. buffer is nullptr");
        std::exit(0);
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        printf("Open file failed. path = %s", filePath.c_str());
        std::exit(0);
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size) {
        printf("Write file Failed.");
        std::exit(0);
        return false;
    }

    return true;
}"""

main_start = """int main(int argc, char *argv[]) {
    aclInit(nullptr);
    uint32_t devCount = 0;
    aclrtGetDeviceCount(&devCount);
    int deviceId = 0;
    if (argc > 1) {
        deviceId = std::stoi(argv[1]);
    }
    aclrtSetDevice(deviceId);
    aclrtContext context;
    aclrtCreateContext(&context, deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC);
"""
