# compile_func.py



## compile_func

### 函数说明  
函数编译接口，可将参数内的函数编译成一个__aicore__上的可执行kernel函数。

### 参数说明  
- func: input, 需要编译的函数。
- global：是一个内置函数，它存储当前需要编译的函数的输入和输出参数的参数名和值。

### 返回值
- 无

### sample0  
```python
@sub_kernel(core_num=M)
def matmul(gm_x, gm_y, gm_out): 
    ...

if __name__ == "__main__":
    gm_x = Tensor("GM", "FP32", [M, K], format="ND", multi_core=False)
    gm_y = Tensor("GM", "FP32", [K, N], format="ND", multi_core=False)
    gm_out = Tensor("GM", "FP32", [M, N], format="ND", multi_core=False)
    compile_func(matmul, globals())(gm_x, gm_y, gm_out)
```


# sub_kernel.py

## sub_kernel

### 函数说明  
kernel函数的装饰函数，主要用来记录kernel函数的参数名称和参数位置等信息。

### 参数说明  
- core_num: input, 自动分核处理的核数量。

### 返回值
- 无

### sample0  
```python
@sub_kernel(core_num=M)
def matmul(gm_x, gm_y, gm_out): 
    ...
```

# compile.py


## compile_kernel

### 函数说明  
编译函数，可将记录下来的所有指令信息编译成一个可执行的Ascend算子。

### 参数说明  
- file_path: input, 生成的Ascend算子文件路径。
- name：input, 生成的Ascend算子的函数名。
- hard_sync：input, bool类型，表示是否需要开启核间同步。

### 返回值
- 无

### sample0  
```python
compile_kernel(f"./matmul.cce", "matmul", hard_sync=True)
```