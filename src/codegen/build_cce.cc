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
#include <tvm/base.h>
#include <dmlc/filesystem.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

#include <cstdlib>
#include <climits>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <regex>
#include <ctime>
#include <algorithm>

#include "tvm.h"
#include "codegen/codegen_cce.h"
#include "runtime/cce/cce_common.h"
#include "runtime/cce/cce_module.h"
#include "contrib/cce_parm/cceconf.h"
#include "codegen/build_common.h"
#include "src/common/util.h"

constexpr int UBUF_SIZE = 256 * 1024;
constexpr int CA_SIZE = 64 * 1024;
constexpr int CB_SIZE = 64 * 1024;
constexpr int CC_SIZE = 256 * 1024;
constexpr int CBUF_SIZE = 1024 * 1024;
constexpr int STATUS_BUFFER_SIZE = 64;
constexpr int GM_SIZE = 1024 * 1024 * 8;

namespace akg {
namespace codegen {
namespace {
class CoreCpuCoproc : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == ktvm::ir::attr::coproc_scope) {
      flag_iscore_ = true;
    } else {
      IRVisitor::Visit_(op);
    }
  }
  bool flag_iscore_{false};
};

bool IsCCECore(const Array<LoweredFunc> &funcs) {
  CoreCpuCoproc visitor;
  for (LoweredFunc f : funcs) {
    if (!f) {
      continue;
    }
    visitor.Visit(f->body);
    if (visitor.flag_iscore_) {
      return true;
    }
  }
  return false;
}
}  // namespace

class CdiffSourceList {
 public:
  static CdiffSourceList *GetInstance() {
    static CdiffSourceList list;
    if (!instance_) {
      instance_ = &list;
    }
    return instance_;
  }
  std::vector<std::string> Get() const { return cdiff_source_list_; }
  void Add(const std::string &source) { cdiff_source_list_.push_back(source); }

 private:
  static CdiffSourceList *instance_;
  std::vector<std::string> cdiff_source_list_;
  CdiffSourceList() {}
  ~CdiffSourceList() {}
};

CdiffSourceList *CdiffSourceList::instance_ = nullptr;

TVM_REGISTER_API("build_cce.GetCdiffSourceList").set_body([](const ktvm::TVMArgs args, ktvm::TVMRetValue *ret) {
  CdiffSourceList *inst = CdiffSourceList::GetInstance();
  std::vector<std::string> sources = inst->Get();
  Array<Expr> res;
  for (auto source : sources) {
    res.push_back(Expr(source));
  }
  *ret = res;
});

void GetTempDir(const std::string &tmp_path, const std::string &target, std::string &tmp_code, std::string &tmp_target,
                std::string &tmp_linked_target) {
  if (target == "cce_core") {
    tmp_code = tmp_path + "/my_kernel_core.cce";
    tmp_target = tmp_path + "/my_kernel_core.o";
  } else if (target == "cce_cpu") {
    tmp_code = tmp_path + "/my_kernel_cpu.cce";
    tmp_target = tmp_path + "/my_kernel_cpu_prelink.o";
    tmp_linked_target = tmp_path + "/my_kernel_cpu.o";
  } else if (target == "cce_cpu_llvm") {
    tmp_code = tmp_path + "/my_kernel_cpu.ll";
    tmp_target = tmp_path + "/my_kernel_cpu_prelink.o";
    tmp_linked_target = tmp_path + "/my_kernel_cpu.o";
  }
}

bool IsInMode(const std::string &mode) {
  CHECK(mode == "csim" || mode == "ccesim" || mode == "cdiff");
  const char *runtime_mode = getenv("RUNTIME_MODE");
  if (runtime_mode != nullptr) {
    return std::string(runtime_mode) == mode;
  }
  return false;
}

void CheckFilename(const std::string &filename) {
  std::regex reg("^[a-zA-Z0-9-_\\.]+$");
  CHECK(std::regex_match(filename, reg));
}

// Generate the code for block_idx iterator in multi-core
std::string GenerateMulticoreBlockidx() {
  if (IsInMode("cdiff")) {
    return "static iterator_t(block_idx, 0);\n";
  }
  return "static size_t block_idx = 0;\n";
}

// Replace for(int i = ...) to for(iterator_t(i) = ...)
void ReplaceIterator2Tracked(std::string &code) {
  std::regex reg("(for\\s*\\(\\s*)(int|int32_t|size_t)\\s+([a-zA-Z0-9_]+)(\\s*=\\s*)([^;]+);");
  code = std::regex_replace(code, reg, "$1iterator_t($3, $5);");
}

// Replace POD types such as int32_t to tracked object types such as int32
std::string ReplaceKernelPointer2TrackedType(std::string code, const std::vector<std::string> &storage_scopes) {
  // replace (__ubuf__ int32_t *) to (__ubuf__ int32 *)
  for (auto storage_scope : storage_scopes) {
    std::string pattern = "(" + storage_scope + "\\s+(int|uint)[0-9]+_t\\s*\\*";
    code = std::regex_replace(code, std::regex(pattern), "$1 *");
  }
  // replace (int32_t)var[index] to (int32)var[index]
  std::string pattern = "\\(((int|uint)(int|uint)(8|16|32))_t\\)(\\s*[0-9A-Za-z_]+\\[.+\\])";
  code = std::regex_replace(code, std::regex(pattern), "($1)$4");
  // replace for(int i = ...) to for(iterator_t(i) = ...)
  ReplaceIterator2Tracked(code);
  return code;
}

// Rewrite constant base addresses of storage scopes to (allocated base of the storage scope + constant base)
// (__ubuf__ half *)(0) -> (__ubuf__ half *)(__ubuf__base + 0)
// (__cbuf__ int32_t *) 128 -> (__cbuf__ int32_t *)(__cbuf__base + 128)
std::string SetBaseAddrForBuffers(const std::string &code_in, const std::map<std::string, int> &storage_scopes) {
  std::string code(code_in);
  for (auto storage_scope : storage_scopes) {
    std::string pattern =
      "(\\(\\s*" + storage_scope.first + "\\s*[a-zA-Z0-9_]+\\s*\\*\\s*\\)[(]*(0x[0-9A-Fa-f]+|[0-9]+)[) ]*";
    code = std::regex_replace(code, std::regex(pattern), "$1(" + storage_scope.first + "_base + $2");
  }
  return code;
}

// Default split by space, results ignore empty string
std::vector<std::string> Split(const std::string &str) {
  std::vector<std::string> res;
  bool has_space(false);
  std::string::const_iterator start = str.begin();
  for (auto it = str.begin(); it != str.end(); ++it) {
    if (isspace(*it)) {
      has_space = true;
      if (start != it) {
        res.push_back(str.substr(std::distance(str.begin(), start), std::distance(start, it) + 1));
      }
      start = it;
      ++start;
    }
  }
  if (!has_space) {
    res.push_back(str);
  }
  return res;
}

std::string Strip(const std::string &str, char ch) {
  for (auto it = str.begin(); it != str.end(); ++it) {
    if (*it != ch) {
      std::string strip_head(it, str.end());
      for (auto rev_it = strip_head.rbegin(); rev_it != strip_head.rend(); ++rev_it) {
        if (*rev_it != ch) {
          return std::string(strip_head.rend() + 1, rev_it + 1);
        }
      }
    }
  }
  return "";
}

// Strip space at front and end
std::string Strip(const std::string &raw_in) {
  std::string raw(raw_in);
  for (auto it = raw.begin(); it != raw.end(); ++it) {
    if (!isspace(*it)) {
      std::string tmp(it, raw.end());
      for (auto rev_it = tmp.rbegin(); rev_it != tmp.rend(); ++rev_it) {
        if (!isspace(*rev_it)) {
          return std::string(tmp.rend() + 1, rev_it + 1);
        }
      }
    }
  }
  return "";
}

// Replace const pointer with a storage scope to dynamic address
// Avoids the CCE code to access unallocated pointer directly
// This code is not general, designed for autotensor generated CCE kernel only
std::string ReplaceStorageScopeConstPointer(const std::string &cce_code, bool allow_kernel_use_gm = false) {
  std::map<std::string, int> storage_scopes;
  storage_scopes["__ubuf__"] = UBUF_SIZE;
  storage_scopes["__ca__"] = CA_SIZE;
  storage_scopes["__cb__"] = CB_SIZE;
  storage_scopes["__cc__"] = CC_SIZE;
  storage_scopes["__cbuf__"] = CBUF_SIZE;
  storage_scopes[""] = STATUS_BUFFER_SIZE;

  if (allow_kernel_use_gm) {
    storage_scopes["__gm__"] = GM_SIZE;
  }
  const std::string alignment("1024");

  CHECK_NE(std::string::npos, cce_code.find("__aicore__")) << "__aicore__ not found in CCE code";
  auto kernel_def_loc = cce_code.find("__aicore__");
  std::string kernel_def = cce_code.substr(kernel_def_loc);
  std::string::size_type kernel_start_loc = kernel_def.find("{");
  CHECK_NE(kernel_start_loc, std::string::npos) << "Cannot determine kernel start location";
  std::string kernel_header = cce_code.substr(0, kernel_def_loc) + kernel_def.substr(0, kernel_start_loc);
  std::string kernel_body = kernel_def.substr(kernel_start_loc);
  std::string kernel_preamble = "\n";

  for (auto item : storage_scopes) {
    auto storage_scope = item.first;
    auto size = item.second;
    kernel_preamble += " " + storage_scope + " uint8 * " + storage_scope + "_base = (uint8 *)aligned_alloc(" +
                       alignment + " * sizeof(uint8), " + std::to_string(size) + " * sizeof(uint8));\n";
    kernel_preamble += " CHECK(" + storage_scope + "_base);\n";
    kernel_preamble += "  CHECK((size_t)" + storage_scope + "_base %% (" + alignment + " * sizeof(uint8)) == 0);\n";
    if (IsInMode("cdiff")) {
      kernel_preamble +=
        "  record_mem_region(\"" + storage_scope + "\", " + storage_scope + "_base, " + std::to_string(size) + ");\n";
    }
    kernel_preamble +=
      "  clear_memory((void *)" + storage_scope + "_base, " + std::to_string(size) + " * sizeof(uint8));\n";
  }
  kernel_preamble += "\n";

  kernel_body = SetBaseAddrForBuffers(kernel_body, storage_scopes);
  std::vector<std::string> pointer_storage_scopes({"__ubuf__", "__ca__", "__cb__", "__cc__", "__cbuf__", "__gm__"});

  if (IsInMode("cdiff")) {
    kernel_body = ReplaceKernelPointer2TrackedType(kernel_body, pointer_storage_scopes);
    kernel_header = ReplaceKernelPointer2TrackedType(kernel_header, pointer_storage_scopes);
  }

  return kernel_header + kernel_preamble + kernel_body;
}

// Generate code to allocate tracked GM tensors
std::string GenerateAllocateTracker(const std::vector<std::string> &type_strings,
                                    const std::vector<std::string> &name_strings) {
  std::string main("\n");
  for (unsigned int param = 0; param < name_strings.size(); ++param) {
    auto len = type_strings[param].length();
    std::string tracked_type = type_strings[param].substr(0, len - 2);
    std::string tracked_name = name_strings[param] + "_tracked";
    main += "  " + tracked_type + " * " + tracked_name + " = (" + tracked_type +
            "*) aligned_alloc(alignment * sizeof(uint8), file_size_)" + std::to_string(param) + " * sizeof(uint8);\n";
    main += "  CHECK(" + tracked_name + ");\n";
    main += "  CHECK((size_t)" + tracked_name + " %% (alignment * sizeof(uint8)) == 0);\n";
    main += "  record_mem_region(\"" + name_strings[param] + "\", " + tracked_name + ", file_size_" +
            std::to_string(param) + ");\n";
    main += "  clear_memory((void *)" + tracked_name + ", file_size_" + std::to_string(param) + " * sizeof(uint8));\n";
  }
  return main + "\n";
}

// Generate data movement code from untracked input tensor to tracked GM tensors
std::string GenerateCopyDataFromTracker(const std::vector<std::string> &name_strings) {
  int num_params = name_strings.size();
  std::string main = "\n";
  for (int param = 0; param < num_params; ++param) {
    std::string tracked_name = name_strings[param] + "_tracked";
    main += "  for (iterator_t(i,0); i < file_size_" + std::to_string(param) + " / sizeof(*" + name_strings[param] +
            "); i++) {\n";
    main += "    " + name_strings[param] + "[i] = " + tracked_name + "[i].GetValue();\n";
    main += "\n";
  }
  return main + "\n";
}
// Generate data movement code from tracked GM tensors to untracked output tensor
std::string GenerateCopyData2Tracker(const std::vector<std::string> &name_strings) {
  std::string main("\n");
  for (unsigned int param = 0; param < name_strings.size(); ++param) {
    std::string tracked_name = name_strings[param] + "_tracked";
    main += "  for (iterator_t(i, 0); i < file_size_" + std::to_string(param) + " / sizeof(*" + name_strings[param] +
            "); i++) {\n";
    main += "    " + tracked_name + "[i] = " + name_strings[param] + "[i];\n";
    main += "  }\n";
  }
  return main + "\n";
}

std::string Join(const std::string &sep, const std::vector<std::string> &strings) {
  CHECK_GT(strings.size(), 1);
  std::string res = strings[0];
  for (unsigned int i = 1; i < strings.size(); ++i) {
    res += sep + strings[i];
  }
  return res;
}

// Generate code to invoke the kernel from main function
std::string GenerateKernelCall(const std::string &kernel_name, const std::vector<std::string> &name_strings) {
  std::vector<std::string> name_strings_tracked;
  for (auto name : name_strings) {
    name_strings_tracked.push_back(name + "_tracked");
  }
  if (IsInMode("cdiff")) {
    return kernel_name + "(" + Join(", ", name_strings_tracked) + ");\n";
  } else {
    return kernel_name + "(" + Join(", ", name_strings) + ");\n";
  }
}

// Generate main function (input, output) for CCE fast CPU simulation
std::string GenerateMainForCsim(const std::string &cce_code, const int block_dim = -1) {
  std::string::size_type func_start_loc = cce_code.find("__aicore__");
  CHECK_NE(std::string::npos, func_start_loc) << "__aicore__ not found in CCE code";

  std::string func_code = cce_code.substr(func_start_loc);
  std::string::size_type open_bracket = func_code.find("(");
  std::string::size_type close_bracket = func_code.find(")");
  bool not_valid(open_bracket == std::string::npos || close_bracket == std::string::npos ||
                 open_bracket >= close_bracket);
  CHECK(!not_valid) << "kernel is not a valid function call";

  std::vector<std::string> func_code_split = Split(func_code.substr(0, open_bracket).c_str());
  CHECK(func_code_split.size());
  std::string kernel_name = func_code_split.back();
  CHECK(!kernel_name.empty()) << "Cannot determine kernel name";

  std::string func_param_str = func_code.substr(open_bracket + 1, close_bracket);
  std::vector<std::string> func_params;
  std::string::size_type loc;
  loc = func_param_str.find(",");
  while (loc != std::string::npos) {
    std::string func_param = Strip(func_param_str.substr(0, loc));
    if (!func_param.empty()) {
      func_params.push_back(func_param);
    }
    func_param_str.erase(0, loc + 1);
    loc = func_param_str.find(",");
  }
  if (!func_param_str.empty()) {
    func_params.push_back(func_param_str);
  }
  CHECK(func_params.size()) << "CCE kernel has params";

  std::vector<std::string> type_strings;
  std::vector<std::string> name_strings;
  for (auto param : func_params) {
    std::vector<std::string> idents_raw = Split(param);
    std::vector<std::string> idents;
    for (auto ident : idents_raw) {
      if (ident != "*" && ident != "__restrict__") {
        idents.push_back(Strip(ident, '*'));
      }
    }
    CHECK_EQ(idents.size(), 3) << "CCE kernel param" << param << " not recognized";
    CHECK_EQ(idents[0], "__gm__") << "CCE kernel param" << param << " is not marked __gm__";
    std::string type_string = idents[1];
    if ('t' != type_string.back()) {
      type_string = type_string + "_t";
    }
    type_strings.push_back(type_string);
    name_strings.push_back(idents[2]);
  }

  std::string main("int main() {\n");
  std::vector<std::string> signals_to_capture(
    {"SIGSEGV", "SIGBUS", "SIGABRT", "SIGINT", "SIGHUP", "SIGPIPE", "SIGSTOP"});
  for (auto signal : signals_to_capture) {
    main += "  signa(" + signal + ", signal_handler);\n";
  }
  main += "\n  const int alignment = 1024;\n  int retval = 0;\n  FILE *fp;\n";
  for (unsigned int i = 0; i < func_params.size(); ++i) {
    std::string param = std::to_string(i);
    main += "\n";
    main += "  fp = fopen(\"in_" + param + ".bin\", \"rb\");\n";
    main += "  CHECK(fp);\n";
    main += "  CHECK(fseek(fp, 0, SEEK_END) == 0);\n";
    main += "  int file_size_" + param + " = ftell(fp);\n";
    main += "  CHECK(file_size_" + param + " > 0);\n";
    main += "  rewind(fp);\n";
    main += "  " + type_strings[i] + " * " + name_strings[i] + " = (" + type_strings[i] +
            " *) aligned_alloc(alignment, file_size_" + param + ");\n";
    main += "  CHECK(" + name_strings[i] + ");\n";
    main += "  retval = fread((void *)" + name_strings[i] + ", 1, file_size_" + param + ", fp);\n";
    main += "  CHECK(retval == file_size_" + param + ")";
    main += "  << \"expected size \" << file_size_" + param + " << \", actual read size \" << retval;\n";
    main += "  fclose(fp);\n";
  }

  if (IsInMode("cdiff")) {
    main += GenerateAllocateTracker(type_strings, name_strings);
    main += GenerateCopyData2Tracker(name_strings);
    main += "\n";
    main += "  DisableUndefinedAssignCheck();\n";
    main += "  launch_kernel();\n";
  }

  std::string kernel_call = GenerateKernelCall(kernel_name, name_strings);
  if (block_dim != -1) {
    main += "  for (block_idx = 0; block_idx < " + std::to_string(block_dim) + "; block_idx++) {\n";
    main += "    " + kernel_call + "  }\n";
  } else {
    main += "  " + kernel_call;
  }

  if (IsInMode("cdiff")) {
    main += "  RestoreUndefinedAssignCheck();\n";
    main += GenerateCopyDataFromTracker(name_strings);
  }

  for (unsigned int param = 0; param < func_params.size(); ++param) {
    main += "\n";
    main += "  fp = fopen(\"out_" + std::to_string(param) + ".bin\", \"wb\");\n";
    main += "  CHECK(fp);\n";
    main += "  retval = fwrite((void *)" + name_strings[param] + ", 1, file_size_" + std::to_string(param) + ", fp);\n";
    main += "  CHECK(retval == file_size_" + std::to_string(param) + ")";
    main +=
      "  << \"expected size \" << file_size_" + std::to_string(param) + " << \", actual write size \" << retval;\n";
    main += "  fclose(fp);\n";
  }

  return main + "}\n";
}

// Mangle CCE code so that GCC can compile it
void MangleCceCode(const std::string &cce_fname, bool need_replace_storage_scope = false, bool need_add_main = false,
                   int block_dim = -1) {
  int ret = access(cce_fname.c_str(), F_OK);
  CHECK_EQ(ret, 0) << "CCE source file " + cce_fname + "not found";

  std::ifstream cce_file(cce_fname);
  CHECK(cce_file.is_open());
  std::stringstream cce_stream;
  cce_stream << cce_file.rdbuf();
  std::string cce(cce_stream.str());
  cce_file.close();

  std::ofstream new_cce_file(cce_fname);
  CHECK(new_cce_file.is_open());
  if (block_dim != -1) {
    new_cce_file << GenerateMulticoreBlockidx();
  }
  if (need_replace_storage_scope) {
    cce = ReplaceStorageScopeConstPointer(cce);
  }
  new_cce_file << cce;
  if (need_add_main) {
    new_cce_file << "\n";
    new_cce_file << GenerateMainForCsim(cce, block_dim);
  }
  new_cce_file.close();
}

// Mangle CCE code to make it able to compile with GCC
std::string CcePostprocCsimMangleCode(const std::string &code, uint32_t block_dim, const std::string &kernel_name) {
  std::string source_filename = kernel_name + ".cce";
  CheckFilename(source_filename);

  std::ofstream source_file(source_filename);
  CHECK(source_file.is_open());
  source_file << code;
  source_file.close();

  MangleCceCode(source_filename, true, true, block_dim);
  return source_filename;
}

std::string Dirname(const std::string &filepath) {
  std::string::size_type pos = filepath.find_last_of("/");
  if (pos == std::string::npos) {
    return filepath;
  }
  return filepath.substr(0, pos);
}

// Traverse parent directories of the current directory and find the full path
// of csim header directory
// Current directory must be inside the project root directory
// relative path is a constant string, please update it if the directory is changed
std::string GetCsimHeaderPath() {
  std::string relative_path("/src/runtime/csim");
  int max_path_depth = 16;

  char cwd[PATH_MAX];
  char *ret = getcwd(cwd, sizeof(cwd));
  CHECK(ret != nullptr);

  char abspath[PATH_MAX];
  char *res = realpath(cwd, abspath);
  CHECK(res != nullptr);
  CHECK_EQ(0, access(abspath, F_OK));

  int path_depth_count = 0;
  std::string dirname(abspath);
  while (access((dirname + relative_path).c_str(), F_OK) != 0) {
    std::string parent_path = Dirname(dirname);
    if (parent_path == dirname) {
      return "";
    }
    dirname = parent_path;
    ++path_depth_count;
    if (path_depth_count > max_path_depth) {
      return "";
    }
  }

  return dirname + relative_path;
}

// Create build directory for csim
std::string MakeCsimDir(const std::string &csim_pass) {
  char cwd[PATH_MAX];
  char *ret = getcwd(cwd, sizeof(cwd));
  CHECK(ret != nullptr);

  std::string csim_base_dir(cwd);
  csim_base_dir = csim_base_dir + "/csim";
  bool exist(0 == access(csim_base_dir.c_str(), F_OK));
  if (!exist) {
    CHECK_EQ(0, mkdir(csim_base_dir.c_str(), 0777));
  }

  std::string csim_dir = csim_base_dir + "/" + csim_pass;
  bool is_exist(0 == access(csim_dir.c_str(), F_OK));
  if (!is_exist) {
    CHECK_EQ(0, mkdir(csim_dir.c_str(), F_OK));
  }

  return csim_dir;
}

// Add #include header_files to source code, and write the code to write_file_name
void AddHeader2File(const std::string &write_file_name, const std::string &source_code,
                    const std::vector<std::string> &header_files) {
  std::ofstream new_source_file(write_file_name);
  CHECK(!new_source_file.is_open());
  for (auto header_file : header_files) {
    new_source_file << "#include \"" << header_file << "\"\n";
  }
  new_source_file << source_code;
  new_source_file.close();
}

void Copyfile(const std::string &src_file, const std::string &dst_file) {
  std::ifstream src(src_file, std::ios::binary);
  std::ofstream dst(dst_file, std::ios::binary);
  CHECK(src.is_open() && dst.is_open());
  dst << src.rdbuf();
  src.close();
  dst.close();
}

// Replace POD types such as int32_t to tracked object types such as int32
void ReplaceLibraryPointer2TrackedType(std::string &code) {
  // replace (int32_t *) to (int32 *)
  code = std::regex_replace(code, std::regex("([^a-zA-Z0-9_](int|uint)(8|16|32))_t\\s*\\*"), "$1 *");
  // replace <int32_t, half> to <int32, half> and <int32_t, int32_t> to <int32, int32>
  code = std::regex_replace(code, std::regex("(<|,)(\\s*(int|uint)(8|16|32))_t\\s*(>|,)"), "$1$2$5");
  code = std::regex_replace(code, std::regex("(<|,)(\\s*(int|uint)(8|16|32))_t\\s*(>|,)"), "$1$2$5");
  // replace sizeof(int32_t) to sizeof(int32)
  code = std::regex_replace(code, std::regex("sizeof\\(\\s*((int|uint)(8|16|32))_t\\s*\\)"), "sizeof($1)");
  // replace for(int i = ...) to for(iterator_t(i) = ...)
  ReplaceIterator2Tracked(code);
  // replace vector_dup(..., int32_t a, ...) to vector_dup(..., int32 a, ...)
  code = std::regex_replace(code, std::regex("([(,]\\s*)((int|uint)(8|16|32))_t\\s+a(\\s*[),])"), "$1$2 $5");
  // replace static to static uint8
  code = std::regex_replace(code, std::regex("static bool "), "static uint8 ");
  // replace <bool> to <uint8>
  code = std::regex_replace(code, std::regex("(<|,)\\s*bool\\s*(>|,)"), "$1uint8$2");
}

// Replace type names in a library file to distinguish POD types and tracked types
void MangleTypeInLibraryFile(const std::string &file_path) {
  std::ifstream f(file_path);
  CHECK(f.is_open());
  std::stringstream buffer;
  buffer << f.rdbuf();
  std::string code(buffer.str());
  f.close();

  ReplaceLibraryPointer2TrackedType(code);

  std::ofstream file(file_path);
  CHECK(file.is_open());
  file << "#include \"compute_tracker.h\"\n\n";
  file << code;
  file.close();
}

// Run a shell commond (only support linux)
void RunCmd(const std::string &cmd) {
  CHECK(!cmd.empty());
  LOG(INFO) << "cmd : " << cmd;
  std::time_t start, end;
  start = time(nullptr);

  FILE *fp;
  fp = popen(cmd.c_str(), "r");
  CHECK(fp != nullptr) << cmd << " error, errno: " << errno;
  if (pclose(fp) == -1) {
    LOG(FATAL) << "pclose error, cmd: " << cmd;
  }

  end = time(nullptr);
  double elapse = std::difftime(end, start);
  LOG(INFO) << "cmd execute complete, elapsed time: " << elapse << " s\n";
}

// Compile C kernel to binary for fast CPU simulation
// csim_pass: the binary file name to be generated
// source_code: C kernel source code (string)
// additional_source_files: additional C kernel source files already in build dir
void CompileCsim(const std::string &csim_pass, const std::string &source_code,
                 const std::vector<std::string> &additional_source_files = {}) {
  CheckFilename(csim_pass);
  for (const auto &additional_source_file : additional_source_files) {
    CheckFilename(additional_source_file);
  }
  std::string csim_dir = MakeCsimDir(csim_pass);
  std::string header_path = GetCsimHeaderPath();
  CHECK(!header_path.empty()) << "csim headers not found";
  std::vector<std::string> header_files{"aicore_fast_sim.h"};
  std::vector<std::string> library_source_files{"aicore_fast_sim.cc"};
  std::vector<std::string> other_required_files{"half_float.h", "halide_intrinsics.h", "aicore_debug_funcs.h"};
  std::string c_compile_options(" -O0 -g -std=c++11");

  if (IsInMode("cdiff")) {
    library_source_files.push_back("compute_tracker.cc");
    header_files.insert(header_files.begin(), "compute_tracker.h");
    c_compile_options += " -DENABLE_CDIFF";
  }

  std::string new_source_file_name = csim_pass + ".cpp";
  std::string new_source_path_name = csim_dir + "/" + new_source_file_name;
  AddHeader2File(new_source_path_name, source_code, header_files);

  for (const auto &source_file : additional_source_files) {
    std::string source_path = csim_dir + "/" + source_file;
    std::ifstream f(source_path);
    CHECK(f.is_open());
    std::stringstream f_stream;
    f_stream << f.rdbuf();
    std::string f_stream_string(f_stream.str());
    AddHeader2File(source_path, f_stream_string, header_files);
    f.close();
    f_stream.str("");
  }
  for (auto filename : header_files) {
    Copyfile(header_path + "/" + filename, csim_dir + "/" + filename);
  }
  for (auto filename : library_source_files) {
    Copyfile(header_path + "/" + filename, csim_dir + "/" + filename);
  }
  for (auto filename : other_required_files) {
    Copyfile(header_path + "/" + filename, csim_dir + "/" + filename);
  }

  if (IsInMode("cdiff")) {
    std::vector<std::string> csim_files{"aicore_fast_sim.cc", "halide_intrinsics.h", "aicore_debug_funcs.h"};
    for (auto csim_file : csim_files) {
      MangleTypeInLibraryFile(csim_dir + "/" + csim_file);
    }
  }

  std::string all_source_files = new_source_file_name;
  for (const auto &additional_source_file : additional_source_files) {
    all_source_files += " " + additional_source_file;
  }
  for (const auto &library_source_file : library_source_files) {
    all_source_files += " " + library_source_file;
  }
  std::string compile_cmd = "g++" + c_compile_options + " -o" + csim_pass + " " + all_source_files;
  LOG(INFO) << "csim compile cmd: " + compile_cmd + "\n";
  compile_cmd = "cd " + csim_dir + " && " + compile_cmd + " &>/dev/null";
  RunCmd(compile_cmd);

  std::string executable_file = csim_dir + "/" + csim_pass;
  int ret = access(executable_file.c_str(), F_OK);
  CHECK_EQ(ret, 0) << "Executable file " + executable_file + " not found";
}

// Compile C kernel to binary for fast CPU simulation
// csim_pass: generate binary file name
// csim_fname: C kernel file path (full path)
void CompileCsimFile(const std::string &csim_pass, const std::string &csim_fname) {
  int access_ret = access(csim_fname.c_str(), F_OK);
  CHECK_NE(access_ret, -1) << "C kernel source file " << csim_fname << " not found";

  std::ifstream source_file(csim_fname);
  CHECK(source_file.is_open());
  std::stringstream buffer;
  buffer << source_file.rdbuf();
  std::string source_code(buffer.str());
  source_file.close();

  CompileCsim(csim_pass, source_code);
}

void CcePostprocCcesim(const std::string &code, uint32_t block_dim, const std::string &kernel_name) {
  if (!IsInMode("ccesim")) {
    return;
  }
  std::string binary_filename = "cce_" + kernel_name;
  std::string source_filename = CcePostprocCsimMangleCode(code, block_dim, kernel_name);
  CompileCsimFile(binary_filename, source_filename);
  CHECK_EQ(0, setenv("CCE_KERNEL-NAME", binary_filename.c_str(), 1));
}

const char VERSION_CCE_ARCH_ES[] = "3.5";

// Build the compile command for aicore op
std::string BuildAicoreCompileCmd(const std::string &src_file, const std::string &dst_file) {
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf != nullptr);
  std::string cce_arch = conf->getCompilerValue("Compiler_arch");

  if (conf->getSection() == VERSION_CCE_ARCH_ES) {
    cce_arch = cce_arch + "-es";
  }

  std::string arch = "cce-aicore-only";
  std::string cce_arch_prefix = "cce-aicore-arch";

  std::string cmd =
    "ccec -c -O2 " + src_file + " --" + cce_arch_prefix + "=" + cce_arch + " --" + arch + " -o " + dst_file;
  return cmd;
}

std::string BuildAicoreCompileCmd(const std::string &src_file, const std::string &dst_file,
                                  const Array<NodeRef> &third_libs) {
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf != nullptr);
  std::string cce_arch = conf->getCompilerValue("Compiler_arch");
  if (conf->getSection() == VERSION_CCE_ARCH_ES) {
    cce_arch = cce_arch + "-es";
  }

  std::string arch = "cce-aicore-only";
  std::string cce_arch_prefix = "cce-aicore-arch";

  std::string cmd;
  if (third_libs.size() > 0) {
    cmd = "ccec -c -O2 -I./feature_lib/include";
    std::string temp_lib;
    for (auto lib_name : third_libs) {
      CHECK(lib_name.as<StringImm>());
      temp_lib = " -include " + lib_name.as<StringImm>()->value + ".h";
      cmd.append(temp_lib);
    }
    cmd.append(" " + src_file + " --" + cce_arch_prefix + "=" + cce_arch + " --" + arch + " -o " + dst_file);
  } else {
    cmd = "ccec -c -O2  " + src_file + " --" + cce_arch_prefix + "=" + cce_arch + " --" + arch + " -o " + dst_file;
  }
  return cmd;
}

// Build the compile command for aicpu op
std::string BuildAicpuCompileCmd(const std::string &target, const std::string &src_file, const std::string &dst_file) {
  const char *tvm_aicpu_include_path = std::getenv("TVM_AICPU_INCLUDE_PATH");
  CHECK(tvm_aicpu_include_path != nullptr) << "the TVM_AICPU_INCLUDE_PATH env var not found, need config";

  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf != nullptr);
  bool aicpu_support_os = conf->getCompilerValue("Compiler_aicpu_support_os") == "true";

  std::string arch = "cce-aicpu-only";
  std::string cce_arch_prefix = "cce-aicpu-arch";

  std::string cmd = "ccec -c -O2 " + src_file + " --" + arch + " --" + cce_arch_prefix + "=cortex-a55 -mcpu=cortex-a55";
  if (target == "cce_cpu_llvm") {
    cmd = cmd + " --target=aarch64-hisilicon-cce -fPIC";
  }
  if (aicpu_support_os && target == "cce_cpu") {
    cmd = cmd + " --cce-aicpu-no-firmware";
  } else if (aicpu_support_os && target == "cce_cpu_llvm") {
    cmd = cmd + " -mllvm -cce-aicpu-no-firmware=true";
  }

  std::string paths(tvm_aicpu_include_path);
  paths = paths + ":";
  while (std::string::size_type pos = paths.find(":") != std::string::npos) {
    std::string path = paths.substr(0, pos);
    if (paths.length() > pos + 1) {
      paths = paths.substr(pos + 1);
    } else {
      paths = "";
    }
    if (!path.empty()) {
      char abspath[PATH_MAX];
      char *res = realpath(path.c_str(), abspath);
      CHECK(res != nullptr);
      int ret = access(abspath, F_OK);
      CHECK_NE(ret, -1) << "The abspath could not be found!";
      cmd = cmd + " -I" + abspath;
    }
  }

  cmd = cmd + " -o " + dst_file;
  return cmd;
}

// split code by separator and return the first or the last separated string
std::string Split(const std::string &code, const std::string &separator, bool reverse = false) {
  if (!reverse) {
    auto end = code.find(separator);
    if (end != std::string::npos) {
      return code.substr(0, end);
    }
  } else {
    auto begin = code.rfind(separator);
    if (begin != std::string::npos && code.length() > begin + separator.length()) {
      return code.substr(begin + separator.length());
    }
  }
  return "";
}

// Getting the lib name of op
std::string GetKernelName(const std::string &code, const std::string &target, const std::string &path_target) {
  std::string kernel_name;

  if (path_target.empty()) {
    std::string sub0 = Split(code, "__kernel");
    kernel_name = Split(sub0, " ", true);
    if (target == "cce_cpu_llvm" && kernel_name.length() > 1) {
      kernel_name = kernel_name.substr(1);
    }
  } else {
    std::string sub1 = Split(path_target, ".");
    kernel_name = Split(sub1, "/", true);
  }

  CHECK(!kernel_name.empty()) << "Getting kernel name failed";
  return kernel_name;
}

// Build the link command for aicpu op
std::string BuildAicpuLinkCmd(const std::string &src_file, const std::string &dst_file, const std::string &lib_name) {
  const char *tvm_aicpu_libray_path = std::getenv("TVM_AICPU_LIBRARY_PATH");
  CHECK(tvm_aicpu_libray_path != nullptr) << "TVM_AICPU_LIBRARY_PATH env var not found";

  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf != nullptr);
  bool aicpu_support_os = conf->getCompilerValue("Compiler_aicpu_support_os") == "true";

  std::string lib_type = "-static";
  if (aicpu_support_os) {
    lib_type = "-shared";
  }

  std::string cmd = "aarch64-elf-ld " + src_file + " " + lib_type + " -ltvm_aicpu -lm -lc";

  // add library search path
  std::string lib_paths = std::string(tvm_aicpu_libray_path) + ":";
  while (unsigned int pos = lib_paths.find(":") != std::string::npos) {
    std::string lib_path = lib_paths.substr(0, pos);
    if (lib_paths.length() > pos + 1) {
      lib_paths = lib_paths.substr(pos + 1);
    } else {
      lib_paths = "";
    }

    if (!lib_path.empty()) {
      char lib_realpath[PATH_MAX];
      char *res = realpath(lib_path.c_str(), lib_realpath);
      CHECK(res != nullptr);
      int ret = access(lib_realpath, F_OK);
      CHECK_NE(ret, -1) << "The lib_realpath could not be found!";

      std::string aicpu_lib_path_raw = lib_path + "/../../../../../toolchain/artifacts/aicpu_lib";
      char aicpu_lib_path[PATH_MAX];
      res = realpath(aicpu_lib_path_raw.c_str(), aicpu_lib_path);
      CHECK(res != nullptr);
      cmd = cmd + " -L" + lib_realpath + " -L" + aicpu_lib_path;
    }
  }

  // add include search path
  char *tvm_aicpu_include_path = std::getenv("TVM_AICPU_INCLUDE_PATH");
  CHECK(tvm_aicpu_include_path != nullptr) << "TVM_AICPU_INCLUDE_PATH env var not found";
  std::string inc_paths = std::string(tvm_aicpu_include_path) + ":";
  while (unsigned int pos = inc_paths.find(":") != std::string::npos) {
    std::string inc_paths_s = inc_paths.substr(0, pos);
    if (inc_paths_s.length() > pos + 1) {
      inc_paths_s = inc_paths_s.substr(pos + 1);
    } else {
      inc_paths_s = "";
    }

    if (!inc_paths_s.empty()) {
      char inc_realpath[PATH_MAX];
      char *res = realpath(inc_paths_s.c_str(), inc_realpath);
      CHECK(res != nullptr);
      int ret = access(inc_realpath, F_OK);
      CHECK_NE(ret, -1) << "The inc_realpath could not be found!";
      cmd = cmd + " -I" + inc_realpath;
    }
  }

  // if aicpu has deployed OS
  if (aicpu_support_os) {
    char *lib_includes = std::getenv("TVM_AICPU_OS_SYSROOT");
    CHECK(lib_includes != nullptr) << "TVM_AICPU_OS_SYSROOT env var not found";
    std::string lib_includes_strip = Strip(lib_includes);
    char lib_inc_realpath[PATH_MAX];
    char *res = realpath(lib_includes_strip.c_str(), lib_inc_realpath);
    CHECK(res != nullptr);
    int ret = access(lib_inc_realpath, F_OK);
    CHECK_NE(ret, -1) << "The lib_inc_realpath could not be found!";
    cmd = cmd + " --sysroot=" + lib_inc_realpath;
    cmd = cmd + " -soname=" + lib_name + ".so";
  }

  cmd = cmd + " -o " + dst_file;
  return cmd;
}

// Build the link command for aicore when calling third_libs
std::string BuildAicoreLinkCmd(const std::string &src_file, const std::string &dst_file,
                               const Array<NodeRef> &lib_names) {
  CHECK(!lib_names.empty()) << "Third_libsNames should not be empty when using aicore link";

  std::string linkcmd = "aicore-elf-ld " + src_file + " ";
  std::string temp_lib;
  for (auto lib_name : lib_names) {
    CHECK(lib_name.as<StringImm>());
    temp_lib = "kernel_meta/" + lib_name.as<StringImm>()->value + ".o";
    if (access(temp_lib.c_str(), R_OK) != 0) {
      std::string temp_src = "feature_lib/src/" + lib_name.as<StringImm>()->value + ".cce";
      std::string compile_cmd = BuildAicoreCompileCmd(temp_src, temp_lib);
      RunCmd(compile_cmd);
    }
    linkcmd.append(temp_lib + " ");
  }
  linkcmd.append("-o " + dst_file);
  return linkcmd;
}

// Binary file to string
std::string BinFile2String(const std::string &path) {
  std::ifstream inputfile(path, std::ios::binary);
  CHECK(inputfile.is_open());
  std::vector<unsigned char> file_data_vec((std::istreambuf_iterator<char>(inputfile)),
                                           std::istreambuf_iterator<char>());
  std::string res(file_data_vec.begin(), file_data_vec.end());
  return res;
}

// Compile cce code with ccec from env
std::string CompileCce(const std::string &code, const std::string &target, std::string path_target,
                       const Array<NodeRef> &third_libs) {
  CHECK(target == "cce_core" || target == "cce_cpu" || target == "cce_cpu_llvm");

  // get temp files which using in compile
  dmlc::TemporaryDirectory temp;
  std::string temp_code, temp_target, temp_linked_target;
  GetTempDir(temp.path, target, temp_code, temp_target, temp_linked_target);

  // save code into a temp file
  std::ofstream outfile(temp_code, std::fstream::out);
  CHECK(outfile.is_open());
  outfile << code;
  outfile.close();

  // compile step, both aicore and aicpu
  std::string file_target = temp_target;
  if (!path_target.empty()) {
    file_target = path_target;
  }
  if (target != "cce_core") {
    file_target = temp_target;
  }

  std::string compile_cmd;
  if (target == "cce_core") {
    if (third_libs.size() == 0) {
      compile_cmd = BuildAicoreCompileCmd(temp_code, file_target);
    } else {
      compile_cmd = BuildAicoreCompileCmd(temp_code, temp_target, third_libs);
    }
  } else {
    compile_cmd = BuildAicpuCompileCmd(target, temp_code, file_target);
  }

  RunCmd(compile_cmd);

  // link step, for aicpu only
  if (target == "cce_cpu" || target == "cce_cpu_llvm") {
    if (path_target.empty()) {
      path_target = temp_linked_target;
    }
    std::string kernel_name = GetKernelName(code, target, path_target);
    std::string link_cmd = BuildAicpuLinkCmd(file_target, path_target, kernel_name);
    RunCmd(link_cmd);
  } else if (target == "cce_core" && third_libs.size() > 0) {
    std::string link_cmd = BuildAicoreLinkCmd(temp_target, file_target, third_libs);
    RunCmd(link_cmd);
  }

  return BinFile2String(file_target);
}

/*
 *Function for putting "lib", kernel_name and ".so" together from code,
 *so that ccec can compile cce file.
 */
std::string TvmCallbackCceCompile(const std::string &code, const Array<NodeRef> &third_libs) {
  const char *runtime_mode = getenv("RUNTIME_MODE");
  if (runtime_mode) {
    std::string rt_mode = runtime_mode;
    if (rt_mode == "csim" || rt_mode == "ccesim" || rt_mode == "cdiff") {
      return "";
    }
  }

  DIR *dir = opendir("kernel_meta");
  if (dir == nullptr) {
    auto ret = mkdir("kernel_meta", S_IRWXG | S_IRWXU);
    CHECK(ret == 0 || (ret == -1 && errno == EEXIST)) << "mkdir kernel_meta failed";
  } else {
    int close_ret = closedir(dir);
    CHECK_EQ(close_ret, 0);
  }

  auto pos_end = code.find("_kernel");
  auto sub1 = code.substr(0, pos_end);
  auto pos_begin = sub1.rfind(" ");
  auto kernel_name = sub1.substr(pos_begin + 1);

  bool is_aicpu = false;
  std::string target = "other";
  if (std::string::npos != code.find("__aicore__")) {
    target = "cce_core";
  } else if (std::string::npos != code.find("__aicpu__")) {
    target = "cce_cpu";
    is_aicpu = true;
  } else if (std::string::npos != code.find("aarch64-hisilicon-cce")) {
    target = "cce_cpu_llvm";
    kernel_name = kernel_name.substr(1);
    is_aicpu = true;
  }

  std::string bin_file_prefix = "";
  std::string bin_file_suffix = ".o";
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf != nullptr);
  bool aicpu_support_os = conf->getCompilerValue("Compiler_aicpu_support_os") == "true";
  if (is_aicpu && aicpu_support_os) {
    bin_file_prefix = "lib";
    bin_file_suffix = ".so";
  }
  // If aicpu has deployed os, the file name will be
  // `lib + kernel_name + .so`. Otherwise, it is `kernel_name.o`
  std::string path_target = "kernel_meta/" + bin_file_prefix + kernel_name + bin_file_suffix;
  if (access(path_target.c_str(), F_OK) == 0) {
    auto ret = std::remove(path_target.c_str());
    CHECK_EQ(ret, 0);
  }
  std::string ccebin = CompileCce(code, target, path_target, third_libs);
  if (chmod(path_target.c_str(), S_IRUSR) == -1) {
    LOG(FATAL) << "modify file to readonly fail!";
  }
  return ccebin;
}

// Add prefix for each function and each statement line in functions
// Assume there is no string literal or comment that includes non-matching brackets in the code
std::string AddPrefixForEachLineInFunc(const std::string &source, const std::string &func_prefix,
                                       const std::string &line_prefix) {
  std::vector<std::string> lines = ktvm::common::Split(source, '\n');
  std::string new_source;
  int nested_scope_level = 0;
  for (const auto &line : lines) {
    bool is_in_func = nested_scope_level > 0;
    std::string line_strip = Strip(line);
    bool is_macro_line = !line_strip.empty() && line_strip[0] == '#';
    bool is_else_line = line_strip.length() >= 4 && line_strip.substr(0, 4) == "else";
    bool is_empty_line = line_strip.empty();
    if (is_in_func && !is_macro_line && !is_else_line && is_empty_line) {
      new_source += line_prefix;
    }
    new_source += line + "\n";

    int original_nested_scope_level = nested_scope_level;
    nested_scope_level -= std::count(line.begin(), line.end(), '}');
    nested_scope_level += std::count(line.begin(), line.end(), '{');
    CHECK_GE(nested_scope_level, 0) << "unrecognized CCE mod: nested scope level < 0";

    if (original_nested_scope_level == 0 && nested_scope_level == 1) {
      new_source += func_prefix + "\n";
    }
  }

  CHECK_NE(nested_scope_level, 0) << "Brackets do not match in CCE mode:\n" << source;
  return new_source;
}

// Mangle cdiff source to record file and line number during execution
// Mangle cdiff source so that the function names of different cdiff files do not conflict
std::string MangleCdiffSource(const std::string &source_file_name, const std::string &suffix) {
  std::ifstream source_file(source_file_name);
  CHECK(source_file.is_open());
  std::stringstream source_stream;
  source_stream << source_file.rdbuf();
  std::string source(source_stream.str());
  source_file.close();
  source = AddPrefixForEachLineInFunc(source, "RECORD_FILE();", "RECORD_LINE();  ");

  std::string res;
  std::string::size_type pos;
  while ((pos = source.find("int main")) != std::string::npos) {
    res += source.substr(0, pos) + "int main_" + suffix;
    source = source.substr(pos + 8);
  }
  return res + source;
}

// Write source_code to the file specified by csim_pass_name and file_name
void WriteMangledCdiffFile(const std::string &csim_pass_name, const std::string &file_name,
                           const std::string &source_code) {
  std::string csim_dir = MakeCsimDir(csim_pass_name);
  std::string file_path = csim_dir + "/" + file_name;
  std::ofstream f(file_path);
  CHECK(f.is_open());
  f << source_code;
  f.close();
  return;
}

// Generate main function for cdiff genereated code
std::string GenerateCdiffMain() {
  std::string source;
  source += "int main_record();\n";
  source += "int main_compare();\n";
  source += "int main();\n";
  source += "  sanity_check();\n";
  source += "  begin_record();\n";
  source += "  main_record();\n";
  source += "  begin_compare();\n";
  source += "  main_compare();\n";
  source += "  final_check();\n";
  source += "  return 0;\n";
  source += "}\n";
  return source;
}

// Compile cdiff files
void CompileCdiff(const std::vector<std::string> &cdiff_file_list) {
  CHECK_EQ(cdiff_file_list.size(), 2) << "cdiff must compare two passes!";
  std::string cdiff_record_src = MangleCdiffSource(cdiff_file_list[0], "record");
  std::string cdiff_compare_src = MangleCdiffSource(cdiff_file_list[1], "compare");

  const std::string csim_pass_name("cdiff");
  const char *dump_c_pass = std::getenv("DUMP_C_PASS");
  CHECK(dump_c_pass != nullptr) << "DUMP_C_PASS must be defined in environ!";
  std::vector<std::string> cdiff_pass_names = ktvm::common::Split(dump_c_pass, ',');
  std::string cdiff_new_record_file_name = cdiff_pass_names[0] + ".cpp";
  WriteMangledCdiffFile(csim_pass_name, cdiff_new_record_file_name, cdiff_record_src);
  std::string cdiff_new_compare_file_name = cdiff_new_compare_file_name[1] + ".cpp";
  WriteMangledCdiffFile(csim_pass_name, cdiff_new_compare_file_name, cdiff_compare_src);
  const std::string cdiff_main = GenerateCdiffMain();
  const std::vector<std::string> additional_source_files({cdiff_new_record_file_name, cdiff_new_compare_file_name});
  CompileCsim(csim_pass_name, cdiff_main, additional_source_files);
}

// Add a source file to cdiff
// At most 2 files can be added, and compilation will automatically start after the second file is added
void CompileCdiffAddSource(const std::string &source_filename) {
  CdiffSourceList *inst = CdiffSourceList::GetInstance();
  inst->Add(source_filename);
  std::vector<std::string> cdiff_source_list = inst->Get();
  if (cdiff_source_list.size() == 2) {
    CompileCdiff(cdiff_source_list);
    for (const auto &source_file : cdiff_source_list) {
      CHECK_EQ(std::remove(source_file.c_str()), 0);
    }
  }
  CHECK_LE(cdiff_source_list.size(), 2) << "cdiff supports at most two DUMP_C_PASS to compare, but found > 2 PASS";
}

void CcePostprocCdiff(const std::string &code, uint32_t block_dim, const std::string &kernel_name) {
  if (!IsInMode("cdiff")) {
    return;
  }

  const char *dump_c_pass = std::getenv("DUMP_C_PASS");
  CHECK(dump_c_pass != nullptr) << "Please set DUMP_C_PASS=record_pass,compare_pass for RUNTIME_MODE is cdiff";

  std::vector<std::string> pass_names = ktvm::common::Split(dump_c_pass, ',');
  for (auto &pass_name : pass_names) {
    for (auto &letter : pass_name) {
      letter = std::tolower(letter);
    }
  }

  CHECK_NE(pass_names.size(), 2) << "RUNTIME_MODE=cdiff must specify two passes: DUMP_C_PASS=record_pass,compare_pass";
  for (auto pass_name : pass_names) {
    if (pass_name == "cce") {
      std::string source_filename = CcePostprocCsimMangleCode(code, block_dim, kernel_name);
      CompileCdiffAddSource(source_filename);
    }
  }
}

ktvm::runtime::Module BuildCCE(const Array<LoweredFunc> &funcs, const Array<NodeRef> &third_libs) {
  using ktvm::runtime::Registry;
  bool output_ssa = false;
  bool iscore = IsCCECore(funcs);
  ktvm::codegen::CodeGenCCE cg;
  cg.Initialize(output_ssa);

  uint32_t block_dim = 1;
  if (!iscore) {
    LOG(FATAL) << "cce not support aicpu !!!";
  }
  cg.tag = false;
  for (LoweredFunc f : funcs) {
    cg.AddFunctionCore(f);
    if (!f || f->thread_axis.empty()) {
      continue;
    }
    CHECK_EQ(f->thread_axis.size(), 1) << "cce only suport 1 block idx !!!";
    auto &axis = *f->thread_axis.begin();
    CHECK_EQ(axis->thread_tag, "blockIdx.x") << "cce only suport blockIdx.x !!!";
    auto node = axis->dom.get();
    if (node != nullptr) {
      CHECK(axis->dom->extent.as<IntImm>());
      block_dim = static_cast<uint32_t>(axis->dom->extent.as<IntImm>()->value);
    }
  }
  std::string code = cg.Finish();

  std::string fmt = "cce";
  std::string ptx;

  ptx = TvmCallbackCceCompile(code, third_libs);
  std::string kernel_name = Split(Split(code, "_kernel"), " ", true);
  CcePostprocCcesim(code, block_dim, kernel_name);
  CcePostprocCdiff(code, block_dim, kernel_name);

  if (const PackedFunc *f = Registry::Get("tvm_callback_cce_postproc")) {
    code = (*f)(code, block_dim).operator std::string();
  }

  return ktvm::runtime::CceModuleCreate(ptx, fmt, ktvm::codegen::ExtractFuncInfo(funcs), code);
}

#ifdef UT_TEST
TVM_REGISTER_API("codegen.build_ccecpu").set_body([](TVMArgs args, TVMRetValue *rv) { *rv = BuildCCE(args[0]); });
#endif

TVM_REGISTER_API("codegen.build_cce").set_body([](const TVMArgs args, TVMRetValue *rv) {
  bool iscore = IsCCECore(args[0]);
  const PackedFunc *func = ktvm::runtime::Registry::Get("codegen.build_ccecpu");
  if (!func || iscore) {
    if (args.size() == 3) {
      *rv = BuildCCE(args[0], args[2]);
    } else {
      Array<NodeRef> list1;
      *rv = BuildCCE(args[0], list1);
    }
  } else {
    (func->body())(args, rv);
  }
});
}  // namespace codegen
}  // namespace akg
