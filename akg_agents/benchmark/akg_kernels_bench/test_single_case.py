import os
import sys
import importlib.util
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Test case timeout (60s)")

def test_single_case(filepath, timeout=60):
    """Test a single Python case file with timeout"""
    print(f"\n=== 开始测试单个文件: {filepath} ===")
    
    # Add the root directory to Python path
    root_dir = os.path.dirname(os.path.abspath(filepath))
    print(f"  根目录: {root_dir}")
    sys.path.insert(0, root_dir)
    print(f"  Python路径已更新")
    
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
    
    # Convert file path to Python module path
    relative_path = os.path.relpath(filepath, root_dir)
    module_path = relative_path.replace('/', '.').replace('\\', '.').replace('.py', '')
    print(f"  相对路径: {relative_path}")
    print(f"  模块路径: {module_path}")
    
    # Set up timeout
    print(f"  设置超时: {timeout}秒")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        print(f"  正在导入模块...")
        
        # Import the module
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
        
        print(f"  正在获取get_inputs函数...")
        get_inputs_func = getattr(module, 'get_inputs')
        print(f"  get_inputs函数获取成功: {get_inputs_func}")
        
        print(f"  正在获取get_init_inputs函数...")
        get_init_inputs_func = getattr(module, 'get_init_inputs')
        print(f"  get_init_inputs函数获取成功: {get_init_inputs_func}")
        
        # Initialize model
        print(f"  正在获取初始化参数...")
        init_params = get_init_inputs_func()
        print(f"  初始化参数: {init_params}")
        
        print(f"  正在初始化模型...")
        # Match the kernel_verify_template.j2 format
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
        print(f"  输入长度: {len(inputs) if isinstance(inputs, list) else 1}")
        
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
        
    except TimeoutError:
        print(f"  === 文件 {relative_path} 测试超时: {timeout}秒 ===")
        return False
    except Exception as e:
        print(f"  === 文件 {relative_path} 测试失败: {str(e)} ===")
        import traceback
        traceback.print_exc()
        return False
    finally:
        signal.alarm(0)  # Cancel the alarm
        print(f"  超时定时器已取消")

if __name__ == "__main__":
    print("=== 静态形状单个用例测试脚本启动 ===")
    
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("用法: python3 test_single_case.py <path_to_case_file.py> [timeout_seconds]")
        print(f"  当前参数数量: {len(sys.argv)}")
        print(f"  参数列表: {sys.argv}")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"  目标文件路径: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  文件不存在: {filepath}")
        sys.exit(1)
    
    print(f"  文件存在，开始检查...")
    
    timeout = 60  # Default timeout
    if len(sys.argv) == 3:
        try:
            timeout = int(sys.argv[2])
            print(f"  自定义超时时间: {timeout}秒")
        except ValueError:
            print(f"  超时时间必须是整数: {sys.argv[2]}")
            sys.exit(1)
    else:
        print(f"  使用默认超时时间: {timeout}秒")
    
    print(f"  开始测试文件: {filepath}")
    print(f"  超时设置: {timeout}秒")
    
    success = test_single_case(filepath, timeout)
    
    print(f"\n=== 单个用例测试完成 ===")
    print(f"  测试结果: {'成功' if success else '失败'}")
    
    sys.exit(0 if success else 1)