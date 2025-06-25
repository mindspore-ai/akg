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

import os
import subprocess
import sys
import time
from typing import List, Dict, Any, Union, Optional
import shutil

def delete_temp_folder():
    current_dir = os.getcwd()
    temp_path = os.path.join(current_dir, 'temp')
    
    if os.path.exists(temp_path) and os.path.isdir(temp_path):
        try:
            shutil.rmtree(temp_path)
            print(f"已成功删除文件夹: {temp_path}")
        except Exception as e:
            print(f"删除文件夹时出现错误: {e}")
    else:
        print(f"文件夹不存在: {temp_path}")

class TestSuiteRunner:
    def __init__(self, test_dirs: Union[str, List[str]] = '.'):
        self.test_dirs = [test_dirs] if isinstance(test_dirs, str) else test_dirs
        self.test_scripts = []
        self.results = []
        self.grouped_results = {}

    def discover_tests(self, pattern: str = '*.py') -> List[str]:
        from fnmatch import fnmatch
        
        for test_dir in self.test_dirs:
            if not os.path.exists(test_dir):
                print(f"警告: 测试目录 '{test_dir}' 不存在，跳过")
                continue
                
            scripts_in_dir = []
            for root, _, files in os.walk(test_dir):
                for file in files:
                    if fnmatch(file, pattern):
                        script_path = os.path.join(root, file)
                        self.test_scripts.append(script_path)
                        scripts_in_dir.append(script_path)
            
            self.grouped_results[test_dir] = {
                'scripts': scripts_in_dir,
                'results': []
            }
        
        return self.test_scripts

    def _run_test_script(self, script_path: str) -> Dict[str, Any]:
        result = {
            'script': script_path,
            'success': False,
            'exit_code': 0,
            'output': '',
            'duration': 0
        }
        
        try:
            start_time = time.time()
            process = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=600
            )
            end_time = time.time()
            result['duration'] = end_time - start_time
            result['exit_code'] = process.returncode
            result['output'] = process.stdout + process.stderr
            result['success'] = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            result['output'] = f"测试脚本执行超时（超过5分钟）"
        except Exception as e:
            result['output'] = f"执行测试脚本时发生错误: {str(e)}"
            
        return result

    def execute_tests_by_group(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.grouped_results:
            self.discover_tests()
            
        for test_dir, group_data in self.grouped_results.items():
            print(f"\n{'='*40}")
            print(f"开始执行目录 '{test_dir}' 中的测试")
            print(f"{'='*40}")
            
            for script in group_data['scripts']:
                result = self._run_test_script(script)
                group_data['results'].append(result)
                self.results.append(result)
                status = "通过" if result['success'] else "失败"
                print(f"[{status}] {script} ({result['duration']:.2f}s)")
            
            self._print_group_summary(test_dir, group_data['results'])
                
        return self.grouped_results

    def _print_group_summary(self, test_dir: str, results: List[Dict[str, Any]]) -> None:
        passed = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"\n目录 '{test_dir}' 测试结果:")
        print(f"总测试数: {total}")
        print(f"通过: {passed}")
        print(f"失败: {total - passed}")

    def print_summary(self) -> None:
        if not self.results:
            print("没有测试结果")
            return
            
        passed = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - passed
        
        print(f"\n{'='*40}")
        print("整体测试执行摘要")
        print(f"{'='*40}")
        print(f"总测试脚本数: {len(self.results)}")
        print(f"测试目录: {', '.join(self.test_dirs)}")
        print(f"通过: {passed}")
        print(f"失败: {failed}")
        
        if failed > 0:
            print("\n失败的测试:")
            for i, result in enumerate(self.results, 1):
                if not result['success']:
                    print(f"  {i}. {result['script']}")
        print(f"{'='*40}")


if __name__ == "__main__":
    runner = TestSuiteRunner(test_dirs=['./fusion', './math', './reduce', './matmul_L0'])
    test_scripts = runner.discover_tests()
    print(f"发现 {len(test_scripts)} 个测试脚本")
    runner.execute_tests_by_group()
    runner.print_summary()
    delete_temp_folder()