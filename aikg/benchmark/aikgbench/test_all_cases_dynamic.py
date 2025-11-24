"""
Dynamic Shape Version of test_all_cases.py

Usage:
  # Test all dynamic shape cases
  python3 test_all_cases_dynamic.py --dynamic-only
  
  # Test all cases (both static and dynamic)
  python3 test_all_cases_dynamic.py
  
  # Test cases listed in a file
  python3 test_all_cases_dynamic.py --file-list <file_list_path>
  
  # Test only dynamic shape cases listed in a file
  python3 test_all_cases_dynamic.py --file-list <file_list_path> --dynamic-only
  
  # Test only new/modified cases (using git status)
  python3 test_all_cases_dynamic.py --new-only
  
  # Test only new/modified dynamic shape cases
  python3 test_all_cases_dynamic.py --new-only --dynamic-only

This script can test both dynamic shape cases (using get_inputs_dyn_list) 
and static shape cases (using get_inputs), including special cases that 
use get_init_inputs_dyn_list.
"""

import os
import sys
import importlib.util
import argparse


def test_case_file(filepath, root_dir):
    """Test a single case file"""
    relative_path = os.path.relpath(filepath, root_dir)
    print(f"\n=== 开始测试文件: {relative_path} ===")
    
    # 设置默认设备（优先GPU，其次NPU，最后CPU）
    print("  正在检测可用设备...")
    import torch
    device = None
    
    # 优先尝试CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"  Using GPU: {torch.cuda.get_device_name(device)}")
    # 其次尝试NPU（华为昇腾）
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        # 检查环境变量中的设备ID
        device_id = os.environ.get('DEVICE_ID', '0')
        os.environ['DEVICE_ID'] = str(device_id)
        device = torch.device("npu")
        torch.npu.manual_seed(0)
        torch.npu.set_device(int(device_id))
        print(f"  Using NPU: device {device_id}")
    # 最后回退到CPU
    else:
        device = torch.device("cpu")
        print(f"  Using CPU (GPU/NPU not available)")
    
    try:
        print(f"  正在导入模块: {relative_path}")
        
        # Import the module
        module_path = relative_path.replace('/', '.').replace('\\\\', '.').replace('.py', '')
        print(f"  模块路径: {module_path}")
        spec = importlib.util.spec_from_file_location(module_path, filepath)
        print(f"  模块spec创建成功")
        module = importlib.util.module_from_spec(spec)
        print(f"  模块对象创建成功")
        spec.loader.exec_module(module)
        print(f"  模块执行成功")
        
        # Get the required functions
        print(f"  正在获取Model类...")
        model_class = getattr(module, 'Model')
        print(f"  Model类获取成功: {model_class}")
        
        # Check if it's a dynamic shape case (has get_inputs_dyn_list) or static shape case (has get_inputs)
        print(f"  检查是否为动态形状用例...")
        is_dynamic = hasattr(module, 'get_inputs_dyn_list')
        if is_dynamic:
            print(f"  检测到动态形状用例")
            get_inputs_func = getattr(module, 'get_inputs_dyn_list')
        else:
            print(f"  检测到静态形状用例")
            get_inputs_func = getattr(module, 'get_inputs')
        print(f"  输入函数获取成功: {get_inputs_func}")
            
        # Check for init inputs functions
        print(f"  检查初始化输入函数...")
        has_init_inputs = hasattr(module, 'get_init_inputs')
        has_init_inputs_dyn_list = hasattr(module, 'get_init_inputs_dyn_list')
        print(f"  有get_init_inputs: {has_init_inputs}")
        print(f"  有get_init_inputs_dyn_list: {has_init_inputs_dyn_list}")
        
        if has_init_inputs:
            print(f"  使用get_init_inputs函数")
            get_init_inputs_func = getattr(module, 'get_init_inputs')
            init_params_list = [get_init_inputs_func()]
            is_init_dynamic = False
        elif has_init_inputs_dyn_list:
            print(f"  使用get_init_inputs_dyn_list函数")
            get_init_inputs_func = getattr(module, 'get_init_inputs_dyn_list')
            init_params_list = get_init_inputs_func()
            is_init_dynamic = True
        else:
            print(f"  没有初始化输入函数，使用空参数")
            init_params_list = [[]]  # Empty init params
            is_init_dynamic = False
        
        print(f"  初始化参数列表: {init_params_list}")
        print(f"  是否为动态初始化: {is_init_dynamic}")
        
        if is_dynamic:
            # Handle dynamic shape cases
            print(f"  处理动态形状用例...")
            print(f"  正在获取动态输入列表...")
            inputs_list = get_inputs_func()
            print(f"  动态输入列表获取成功，长度: {len(inputs_list)}")
            
            # Determine number of test cases
            num_cases = len(inputs_list)
            
            # If init params are also dynamic, they should match the number of cases
            if is_init_dynamic and len(init_params_list) != num_cases:
                print(f"  FAILED: Mismatch between number of input cases ({num_cases}) and init param cases ({len(init_params_list)})")
                return False
                
            # Test each case
            for i in range(num_cases):
                print(f"    正在测试用例 {i+1}/{num_cases}...")
                inputs = inputs_list[i]
                print(f"    用例 {i+1} 输入: {type(inputs)}")
                init_params = init_params_list[0] if not is_init_dynamic else init_params_list[i]
                print(f"    用例 {i+1} 初始化参数: {init_params}")
                
                # Initialize model for each case if init params are dynamic, or once if static
                if is_init_dynamic or i == 0:
                    print(f"    正在初始化模型...")
                    if isinstance(init_params, list):
                        model = model_class(*init_params)
                    else:
                        model = model_class()
                    print(f"    模型初始化成功: {type(model)}")
                    
                    # Move model to device if available
                    if device.type != "cpu":
                        print(f"    正在将模型移动到设备: {device}")
                        model = model.to(device)
                        print(f"    模型移动成功")
                
                # Move inputs to device and run the model
                print(f"    正在处理输入数据...")
                if isinstance(inputs, list):
                    print(f"    输入是列表，长度: {len(inputs)}")
                    device_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
                    print(f"    输入数据移动到设备完成")
                    print(f"    正在运行模型...")
                    output = model(*device_inputs)
                else:
                    print(f"    输入是单个张量")
                    device_inputs = inputs.to(device) if isinstance(inputs, torch.Tensor) else inputs
                    print(f"    输入数据移动到设备完成")
                    print(f"    正在运行模型...")
                    output = model(device_inputs)
                    
                print(f"    用例 {i+1} PASSED")
        else:
            # Handle static shape cases (original behavior)
            print(f"  处理静态形状用例...")
            # Initialize model
            print(f"  正在初始化模型...")
            init_params = init_params_list[0] if init_params_list else []
            if isinstance(init_params, list):
                model = model_class(*init_params)
            else:
                model = model_class()
            print(f"  模型初始化成功: {type(model)}")
            
            # Move model to device if available
            if device.type != "cpu":
                print(f"  正在将模型移动到设备: {device}")
                model = model.to(device)
                print(f"  模型移动成功")
                
            # Get inputs
            print(f"  正在获取输入数据...")
            inputs = get_inputs_func()
            print(f"  输入数据获取成功: {type(inputs)}")
            
            # Move inputs to device and run the model
            print(f"  正在处理输入数据...")
            if isinstance(inputs, list):
                print(f"  输入是列表，长度: {len(inputs)}")
                device_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
                print(f"  输入数据移动到设备完成")
                print(f"  正在运行模型...")
                output = model(*device_inputs)
            else:
                print(f"  输入是单个张量")
                device_inputs = inputs.to(device) if isinstance(inputs, torch.Tensor) else inputs
                print(f"  输入数据移动到设备完成")
                print(f"  正在运行模型...")
                output = model(device_inputs)
        
        print(f"  === 文件 {relative_path} 测试完成: PASSED ===")
        return True
        
    except Exception as e:
        print(f"  === 文件 {relative_path} 测试失败: {str(e)} ===")
        import traceback
        traceback.print_exc()
        return False


def test_all_cases(root_dir, dynamic_only=False):
    """Test all Python case files in the aikgbench directory"""
    print(f"\n=== 开始测试所有动态形状用例 ===")
    print(f"  根目录: {root_dir}")
    
    # Add the root directory to Python path
    sys.path.insert(0, root_dir)
    print(f"  Python路径已更新")
    
    # Counter for test results
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    failed_files = []
    
    print(f"  开始遍历目录查找测试文件...")
    
    # Walk through all directories to find Python files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"    正在检查目录: {dirpath}")
        
        # 只测试dynamic_shape目录中的文件
        if 'dynamic_shape' not in dirpath:
            print(f"      跳过非dynamic_shape目录")
            continue
            
        print(f"      发现dynamic_shape目录，检查Python文件...")
        
        for filename in filenames:
            # Skip test scripts and any files with "test" in the name
            if (filename.endswith('.py') and 
                filename != 'test_all_cases.py' and 
                filename != 'test_all_cases_dynamic.py' and
                filename != 'test_single_case.py' and
                filename != 'test_single_case_dynamic.py' and
                'test_' not in filename):
                
                filepath = os.path.join(dirpath, filename)
                print(f"      发现测试文件: {filename}")
                
                total_tests += 1
                print(f"      开始测试文件 {total_tests}: {filename}")
                if test_case_file(filepath, root_dir):
                    passed_tests += 1
                    print(f"      文件 {filename} 测试通过")
                else:
                    failed_tests += 1
                    relative_path = os.path.relpath(filepath, root_dir)
                    failed_files.append(relative_path)
                    print(f"      文件 {filename} 测试失败")
            else:
                print(f"      跳过文件: {filename} (测试脚本或包含'test'关键字)")
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY (DYNAMIC SHAPE CASES ONLY)")
    print("="*50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    return failed_tests == 0


def test_cases_from_file(file_list_path, root_dir, dynamic_only=False):
    """Test cases listed in a file"""
    print(f"\n=== 开始从文件列表测试用例 ===")
    print(f"  文件列表路径: {file_list_path}")
    print(f"  根目录: {root_dir}")
    
    # Add the root directory to Python path
    sys.path.insert(0, root_dir)
    print(f"  Python路径已更新")
    
    # Read the file list
    print(f"  正在读取文件列表...")
    with open(file_list_path, 'r') as f:
        case_files = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    
    print(f"  文件列表读取完成，共 {len(case_files)} 个文件")
    
    # Counter for test results
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    failed_files = []
    
    print(f"  开始测试文件列表中的用例...")
    
    # Test each case
    for i, case_file in enumerate(case_files):
        print(f"    正在处理第 {i+1}/{len(case_files)} 个文件: {case_file}")
        
        # 只测试dynamic_shape目录中的文件
        if 'dynamic_shape' not in case_file:
            print(f"      跳过非dynamic_shape文件")
            continue
            
        filepath = os.path.join(root_dir, case_file)
        if os.path.exists(filepath):
            print(f"      文件存在，开始测试...")
            total_tests += 1
            if test_case_file(filepath, root_dir):
                passed_tests += 1
                print(f"      文件 {case_file} 测试通过")
            else:
                failed_tests += 1
                failed_files.append(case_file)
                print(f"      文件 {case_file} 测试失败")
        else:
            print(f"      文件不存在: {filepath}")
            failed_tests += 1
            failed_files.append(case_file)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY (DYNAMIC SHAPE CASES ONLY)")
    print("="*50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    return failed_tests == 0


def test_new_cases(root_dir, dynamic_only=False):
    """Test only new/modified cases by checking git status"""
    print(f"\n=== 开始测试新修改的用例 ===")
    print(f"  根目录: {root_dir}")
    
    # Add the root directory to Python path
    sys.path.insert(0, root_dir)
    print(f"  Python路径已更新")
    
    try:
        # Get list of modified/new Python files
        print(f"  正在检查git状态...")
        import subprocess
        result = subprocess.run(['git', 'status', '--porcelain', '*.py'], 
                               cwd=root_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  git状态检查失败，返回码: {result.returncode}")
            print(f"  错误输出: {result.stderr}")
            return False
            
        print(f"  git状态检查成功")
        lines = result.stdout.strip().split('\n')
        print(f"  原始git输出行数: {len(lines)}")
        
        modified_files = []
        
        for i, line in enumerate(lines):
            if line.strip():
                print(f"    解析第 {i+1} 行: {line}")
                # Parse git status output
                # Example: " M static_shape/norm/LayerNorm_001.py"
                parts = line.split()
                if len(parts) >= 2:
                    filepath = parts[1] if parts[0] in ['M', 'A', '??'] else parts[0]
                    print(f"      解析文件路径: {filepath}")
                    # Only include Python files in dynamic_shape directory
                    if filepath.endswith('.py') and 'dynamic_shape' in filepath:
                        modified_files.append(filepath)
                        print(f"      添加到修改文件列表: {filepath}")
                    else:
                        print(f"      跳过文件: {filepath} (非Python文件或不在dynamic_shape目录)")
                else:
                    print(f"      跳过无效行: {line}")
        
        if not modified_files:
            print("  没有发现修改/新增的Python文件")
            return True
            
        print(f"  发现 {len(modified_files)} 个修改/新增的Python文件:")
        for f in modified_files:
            print(f"    - {f}")
            
        # Counter for test results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        failed_files = []
        
        print(f"  开始测试修改的文件...")
        
        # Test each modified file
        for i, case_file in enumerate(modified_files):
            print(f"    正在测试第 {i+1}/{len(modified_files)} 个文件: {case_file}")
            filepath = os.path.join(root_dir, case_file)
            if os.path.exists(filepath):
                print(f"      文件存在，开始测试...")
                total_tests += 1
                if test_case_file(filepath, root_dir):
                    passed_tests += 1
                    print(f"      文件 {case_file} 测试通过")
                else:
                    failed_tests += 1
                    failed_files.append(case_file)
                    print(f"      文件 {case_file} 测试失败")
            else:
                print(f"      文件不存在: {filepath}")
                failed_tests += 1
                failed_files.append(case_file)
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY (DYNAMIC SHAPE CASES ONLY)")
        print("="*50)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        
        if failed_files:
            print("\nFailed files:")
            for file in failed_files:
                print(f"  - {file}")
        
        return failed_tests == 0
        
    except Exception as e:
        print(f"  检查git状态时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== 动态形状测试脚本启动 ===")
    
    # Get the root directory (assuming script is in the root)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"  脚本路径: {__file__}")
    print(f"  根目录: {root_dir}")
    
    # Parse command line arguments
    print(f"  正在解析命令行参数...")
    parser = argparse.ArgumentParser(description='Test aikgbench dynamic shape cases')
    parser.add_argument('--file-list', '-f', help='Test cases listed in a file')
    parser.add_argument('--new-only', '-n', action='store_true', help='Test only new/modified cases (using git status)')
    args = parser.parse_args()
    
    print(f"  命令行参数解析完成:")
    print(f"    --file-list: {args.file_list}")
    print(f"    --new-only: {args.new_only}")
    
    if args.file_list:
        # Test cases from file list
        print(f"  选择测试模式: 从文件列表测试")
        success = test_cases_from_file(args.file_list, root_dir)
    elif args.new_only:
        # Test only new/modified cases
        print(f"  选择测试模式: 仅测试新修改的用例")
        success = test_new_cases(root_dir)
    else:
        # Test all dynamic shape cases
        print(f"  选择测试模式: 测试所有动态形状用例")
        print(f"  测试目录: {root_dir}")
        success = test_all_cases(root_dir)
    
    print(f"\n=== 测试脚本执行完成 ===")
    print(f"  最终结果: {'成功' if success else '失败'}")
    
    sys.exit(0 if success else 1)