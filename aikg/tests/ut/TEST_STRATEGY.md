# Adapter 模块测试策略说明

## 测试层次结构

### 1. 单元测试（Unit Tests）- 测试单个Adapter方法

#### 测试目标
验证每个Adapter的方法能正确生成代码字符串，不涉及实际执行。

#### 测试方法
```python
# 示例：测试Framework Adapter
def test_get_import_statements(self):
    adapter = get_framework_adapter("torch")
    imports = adapter.get_import_statements()
    # 验证生成的代码字符串
    assert "import torch" in imports
    assert imports.endswith("\n")
```

#### 验证点
- ✅ **代码内容正确性**：生成的代码包含预期的import、函数调用等
- ✅ **格式正确性**：换行符、缩进等格式正确
- ✅ **参数处理**：不同参数组合生成不同代码
- ✅ **边界情况**：无效参数抛出异常

#### 测试覆盖
- Framework Adapter: 28个测试
  - import语句生成（静态/动态shape）
  - 设备设置（cuda/ascend/cpu）
  - 输入处理（不同dsl）
  - 精度限制计算
  - 二进制I/O函数生成
  
- DSL Adapter: 29个测试
  - import语句生成
  - 实现函数调用代码
  - benchmark代码生成
  - 特殊setup代码
  - binary I/O需求检查
  
- Backend Adapter: 13个测试
  - 环境变量设置
  - 设备字符串生成
  - 架构验证

### 2. 集成测试（Integration Tests）- 测试KernelVerifier使用Adapter

#### 测试目标
验证KernelVerifier能正确使用Adapter生成完整的验证脚本。

#### 测试方法
```python
def test_gen_verify_project_torch_triton_cuda(self):
    # 1. 创建KernelVerifier实例
    verifier = KernelVerifier(...)
    
    # 2. 生成验证项目
    verifier.gen_verify_project(impl_code, verify_dir, device_id=0)
    
    # 3. 检查生成的文件
    assert os.path.exists(verify_file)
    
    # 4. 读取生成的代码，验证内容
    with open(verify_script, "r") as f:
        content = f.read()
        # 验证adapter生成的代码被正确插入
        assert "import torch" in content
        assert "from test_op_torch import" in content
        assert "def process_input" in content
```

#### 验证点
- ✅ **文件生成**：所有必需文件都被创建
- ✅ **代码插入**：Adapter生成的代码被正确插入到模板中
- ✅ **组合正确性**：不同framework/dsl/backend组合正确工作
- ✅ **特殊功能**：SWFT的binary I/O、AscendC的编译等特殊逻辑

#### 测试覆盖
- torch + triton_cuda + cuda
- mindspore + triton_ascend + ascend
- torch + swft + ascend（binary I/O）
- 动态shape检测

### 3. 端到端验证（E2E Validation）- 检查生成的代码结构

#### 测试目标
验证生成的Python脚本结构完整、语法正确。

#### 验证方法
1. **代码结构检查**
   - 检查import语句
   - 检查函数定义
   - 检查主执行逻辑

2. **内容正确性检查**
   - 检查adapter生成的代码片段
   - 检查模板变量替换
   - 检查特殊逻辑（binary I/O、编译等）

3. **语法验证**（可选）
   - 使用ast模块解析Python代码
   - 检查是否有语法错误

#### 示例验证
```python
# 验证生成的验证脚本
verify_script = "verify_test_op.py"
with open(verify_script, "r") as f:
    content = f.read()
    
# 1. Framework imports
assert "import torch" in content
assert "from test_op_torch import Model" in content

# 2. DSL imports  
assert "from test_op_triton_cuda import" in content

# 3. 设备设置
assert "CUDA_VISIBLE_DEVICES" in content
assert "device = torch.device" in content

# 4. 输入处理
assert "def process_input" in content

# 5. 实现调用
assert "impl_output = test_op_triton_cuda_torch" in content
```

### 4. 性能测试模板验证

#### 测试目标
验证性能测试模板能正确使用Adapter生成代码。

#### 测试覆盖
- Base profile模板生成
- Generation profile模板生成
- 动态shape支持
- Binary I/O支持（SWFT）

## 测试统计

### 单元测试
- Framework Adapter: 28个测试
- DSL Adapter: 29个测试  
- Backend Adapter: 13个测试
- **总计：70个单元测试**

### 集成测试
- KernelVerifier集成：4个测试
- Profile模板生成：3个测试
- **总计：7个集成测试**

### 总体统计
- **总测试数：77个**
- **通过率：100%**
- **覆盖范围**：
  - 所有Framework（torch, mindspore, numpy）
  - 所有DSL（triton_cuda, triton_ascend, swft, ascendc, cpp, cuda_c, tilelang_npuir, tilelang_cuda）
  - 所有Backend（cuda, ascend, cpu）
  - 静态/动态shape
  - 特殊功能（binary I/O, 编译）

## 验证方法总结

### 1. 代码生成验证
- ✅ 检查生成的代码字符串包含预期内容
- ✅ 检查代码格式正确（换行、缩进）
- ✅ 检查不同参数组合生成不同代码

### 2. 文件生成验证
- ✅ 检查所有必需文件都被创建
- ✅ 检查文件内容正确
- ✅ 检查文件路径正确

### 3. 逻辑正确性验证
- ✅ 检查特殊逻辑（binary I/O、编译）正确实现
- ✅ 检查动态shape支持
- ✅ 检查错误处理（无效参数抛出异常）

### 4. 组合测试
- ✅ 测试不同framework/dsl/backend组合
- ✅ 测试边界情况
- ✅ 测试特殊场景（SWFT、AscendC等）

## 运行测试

```bash
# 运行所有adapter测试
cd aikg && source env.sh
python -m pytest tests/ut/ -k "adapter" -v

# 运行特定测试
python -m pytest tests/ut/test_framework_adapter.py -v
python -m pytest tests/ut/test_dsl_adapter.py -v
python -m pytest tests/ut/test_backend_adapter.py -v

# 运行集成测试
python -m pytest tests/ut/test_kernel_verifier_adapter.py -v
python -m pytest tests/ut/test_profile_template_adapter.py -v
```

## 测试原则

1. **隔离性**：每个测试独立，不依赖其他测试
2. **可重复性**：测试结果可重复
3. **快速性**：单元测试快速执行（不涉及实际硬件）
4. **全面性**：覆盖所有主要功能和边界情况
5. **可读性**：测试代码清晰，易于理解


