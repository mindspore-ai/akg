# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = 'mfusion'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = ['.mlir']

# excludes: A list of directories to exclude from the testsuite.
config.excludes = ['CMakeLists.txt', 'README.txt', 'LICENSE.txt', 'lit.cfg.py', 'lit.site.cfg.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.mfusion_test_build_dir

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.llvm_tools_dir,
    config.mfusion_tools_dir
]
tool_names = [
    'mfusion-opt'
]
tools = [ToolSubst(s, unresolved='ignore') for s in tool_names]
llvm_config.add_tool_substitutions(tools, tool_dirs)

config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

