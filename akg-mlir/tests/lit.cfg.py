# -*- Python -*-

import os
import sys
import re
import platform
import subprocess

import lit.util
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = 'akg'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = ['.mlir']

# excludes: A list of directories to exclude from the testsuite.
config.excludes = ['mlir_files','tmp_files']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.akg_test_build_dir

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.llvm_tools_dir,
    config.akg_tools_dir
]
tool_names = [
    'akg-opt'
]
tools = [ToolSubst(s, unresolved='ignore') for s in tool_names]
tools.extend([ToolSubst('%mlir_lib_dir', config.mlir_lib_dir, unresolved='ignore')])
llvm_config.add_tool_substitutions(tools, tool_dirs)

# where to find the thrid_party dir
config.substitutions.append(('%third_party_path',
                            os.path.join(config.test_source_root, '..', 'third_party')))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# for PTX tests
#llvm_config.with_system_environment(['CUDA_HOME'])

