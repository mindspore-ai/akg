/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef RUNTIME_CSIM_AICORE_DEBUG_FUNCS_H_
#define RUNTIME_CSIM_AICORE_DEBUG_FUNCS_H_

#include "aicore_fast_sim.h"
#include <string.h>
#include <execinfo.h>
#include <signal.h>

#define MAX_STACK_DEPTH 256
// logging
#ifndef CHECK

class ReportFail {
 public:
  ReportFail(const char *file, int line, const char *assertion) {
    std::cerr << file << ":" << line << ": CHECK(" << assertion << ") failed: ";
  }

  std::ostream &GetStream() {
    return std::cerr;
  }

  ~ReportFail() {
    std::cerr << std::endl << std::endl;
    abort();
  }
};

#define CHECK(x) \
  if (!(x)) ReportFail(__FILE__, __LINE__, (#x)).GetStream()

#endif  // #ifdef CHECK

#ifndef CHECK_EQ

#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_LT(x, y) CHECK((x) <  (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_GT(x, y) CHECK((x) >  (y))

#endif  // #ifdef CHECK_EQ

#ifndef LOG

class PrintLog {
 public:
  PrintLog(const char *file, const int line) {
    std::cerr << file << ":" << line << ": ";
  }

  std::ostream &GetStream() {
    return std::cerr;
  }

  ~PrintLog() {
    std::cerr << std::endl;
  }
};

#define LOG(x) PrintLog(__FILE__, __LINE__).GetStream() << (x)
#define INFO "[INFO] "
#define WARNING "[WARNING] "
#define ERROR "[ERROR] "

#endif  // #ifdef LOG

static inline uint64_t GetStatus() {
  return 0;
}

static inline void ClearMemory(void *addr, size_t size) {
  unsigned char *dst = reinterpret_cast<unsigned char *>(addr);
  for (size_t i = 0; i < size; ++i) {
    dst[i] = 0;
  }
}

static void PrintSignalStr(int sig) {
  const char *str = strsignal(sig);
  if (str == nullptr) {
    fprintf(stderr, "Unknown signal %d\n", sig);
  } else {
    fprintf(stderr, "Caught signal %s\n", str);
  }
}

static void SignalHandler(int sig) {
  void *bt_array[MAX_STACK_DEPTH];
  size_t size = backtrace(bt_array, MAX_STACK_DEPTH);
  PrintSignalStr(sig);
  backtrace_symbols_fd(bt_array, size, 2);

  exit(1);
}

static void WrappedRead(void *buffer, const size_t size, FILE *fp) {
  size_t total_read_size = 0;
  unsigned char *buffer_ptr = reinterpret_cast<unsigned char *>(buffer);
  while (total_read_size < size) {
    int retval = fread(buffer_ptr + total_read_size, 1, size - total_read_size, fp);
    CHECK_GT(retval, 0) << "failed to read file: actual read size "
                        << total_read_size << ", expected read size " << size;
    total_read_size += retval;
  }
  CHECK_EQ(total_read_size, size);
}

static void WrappedWrite(void *buffer, const size_t size, FILE *fp) {
  size_t total_write_size = 0;
  unsigned char *buffer_ptr = reinterpret_cast<unsigned char *>(buffer);
  while (total_write_size < size) {
    int retval = fwrite(buffer_ptr + total_write_size, 1, size - total_write_size, fp);
    CHECK_GT(retval, 0) << "failed to read file: actual write size " << total_write_size << ", expected write size "
                      << size;
    total_write_size += retval;
  }
  CHECK_EQ(total_write_size, size);
}

#endif  // RUNTIME_CSIM_AICORE_DEBUG_FUNCS_H_
