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

#include <sys/time.h>
#include <dmlc/logging.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <iomanip>

#define LOG_IN_FILE false
#if LOG_IN_FILE
#include <fstream>
std::ofstream akg_out_file("out.txt");
extern char *__progname;
#endif

std::string getAKGTime() {
  struct timeval tv;
  struct tm now_time;
  std::ostringstream akg_time;
  if (!gettimeofday(&tv, nullptr)) {
    if (localtime_r(&tv.tv_sec, &now_time)) {
      akg_time << now_time.tm_year + 1900 << "-";
      akg_time << std::setw(2) << std::setfill('0') << now_time.tm_mon + 1 << "-";
      akg_time << std::setw(2) << std::setfill('0') << now_time.tm_mday << "-";
      akg_time << std::setw(2) << std::setfill('0') << now_time.tm_hour << ":";
      akg_time << std::setw(2) << std::setfill('0') << now_time.tm_min << ":";
      akg_time << std::setw(2) << std::setfill('0') << now_time.tm_sec << ".";
      akg_time << std::setw(3) << std::setfill('0') << tv.tv_usec / 1000 << ".";
      akg_time << std::setw(3) << std::setfill('0') << tv.tv_usec % 1000;
    }
  }
  return akg_time.str();
}
void AKGLOG(const std::string &msg_info) {
  std::ostringstream log_txt;
  std::string level;
  std::string file_name;
  std::string line;
  std::string module;
  std::string info;
  constexpr int MAX_TXT_LEN = 512 * 8;
  if (msg_info.find(" ") != std::string::npos) {
    auto m1 = msg_info.find(" ");
    level = msg_info.substr(0, m1);
    if (msg_info.find(": ", m1) != std::string::npos) {
      auto m2 = msg_info.find(": ", m1);
      CHECK_GE(msg_info.size(), m1 + 1);
      auto file_str = msg_info.substr(m1 + 1, m2 - m1 - 1);
      CHECK_GE(msg_info.size(), m2 + 2);
      info = msg_info.substr(m2 + 2, msg_info.size() - m2 - 2);
      if (info.size() > MAX_TXT_LEN) {
        info.erase(MAX_TXT_LEN);
      }
      if (info.back() == '\n') {
        info.erase(info.size() - 1);
      }
      if (file_str.find(":") != std::string::npos) {
        auto pos = file_str.find_last_of(":");
        CHECK_GE(file_str.size(), pos + 1);
        line = file_str.substr(pos + 1, file_str.size() - pos - 1);
        file_str.erase(pos);
      }
      if (file_str.find("/") != std::string::npos) {
        auto pos = file_str.find_last_of("/");
        CHECK_GE(file_str.size(), pos + 1);
        file_name = file_str.substr(pos + 1, file_str.size() - pos - 1);
        file_str.erase(pos);
      }
      if (file_str.find("/") != std::string::npos) {
        auto pos = file_str.find_last_of("/");
        CHECK_GE(file_str.size(), pos + 1);
        module = file_str.substr(pos + 1, file_str.size() - pos - 1);
      }
    }
  }
  log_txt << "[" << level << "] AKG";
#if LOG_IN_FILE
  log_txt << "(" << getpid() << "," << __progname << ")";
#endif
  log_txt << ":" << getAKGTime() << " [" << file_name << ":" << line << "] [" << module << "] ";
#if LOG_IN_FILE
  akg_out_file << log_txt.str() << info << "\n";
#else
  std::cout << log_txt.str() << info << "\n";
#endif
}
void FatalLog(std::string msg_error) {
  if (msg_error.find(" ") != std::string::npos) {
    auto m1 = msg_error.find(" ");
    if (msg_error.find(" ", m1) != std::string::npos) {
      auto m2 = msg_error.find(" ", m1 + 1);
      msg_error.erase(m1, m2 - m1);
      AKGLOG(msg_error);
    }
  }
}

#ifdef USE_AKG_LOG
void dmlc::CustomLogMessage::Log(const std::string &msg) {
  if (msg.find("ERROR") == 0) {
    FatalLog(msg);
  } else {
    AKGLOG(msg);
  }
}
#else
void dmlc::CustomLogMessage::Log(const std::string &msg) {
  if (msg.find("ERROR") == 0) {
    FatalLog(msg);
  }
}
#endif
