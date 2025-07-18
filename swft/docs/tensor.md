# tensor.py


## __init__(self, *args, **kwdargs)

### 函数说明  
SWFT Tensor创建函数，其中支持两种创建方式。
方式1：直接从numpy.ndarray创建Tensor；方式2：通过指定Tensor的`mem_type`,`dtype`,`shape`,`format`来创建Tensor.

### 参数说明  
- `ndarray`：位置0参数，类型为numpy.ndarray，如果使用方式1创建Tensor，则该参数是必选参数。
- `memtype`: 位置0参数，类型为字符串，表示Tensor的存放位置 。支持的值：`"GM"`,`"UB"`,`"L1"`,`"L0A"`,`"L0B"`。其中`"GM"`表示Tensor存放在昇腾芯片的全局内存上，`"UB"`,`"L1"`,`"L0A"`,`"L0B"`表示Tensor存放在昇腾芯片的各级高速缓存上。如果使用方式2创建Tensor，则该参数式必选参数。在SWFT Kerne之外的Tensor创建，只支持memtype="GM",因为所有SWFT的计算都是从全局内存的数据搬运开始的。
- `dtype`: 位置1参数，类型为字符串，表示Tensor的数据类型。支持的值：`"FP16"`,`"FP32"`,`"INT8"`,`"INT16"`,`"INT32"`。如果使用方式2创建Tensor，则该参数式必选参数。
- `shape`：位置2参数，类型为list(int)，表示Tensor的形状。如果使用方式2创建Tensor，则该参数式必选参数。
- `format`：位置3参数，类型为字符串，表示Tensor的数据排布格式。支持的值：`"ND"`,`"NZ"`。该参数为可选参数，默认为`"ND"`。
- `multi_core`：位置4参数，类型为bool，表示Tensor在进行算子编译时，使用自动切分编译模式（`multi_core` = `True`）还是手动切分编译模式（`multi_core` = `False`），默认为`False`。关于自动切分模式和手动切分模式，参考[进阶教程.md](进阶教程.md)

### 返回值
- out: output, 返回一个SWFT Tensor, 用来启动算子编译或算子执行。

### sample0  
```python
tensor = Tensor("GM", "FP16", [512], format="ND", multi_core=True) #创建一个Tensor，存放在Global Memory上，数据类型是"FP16"，shape是[512]，存储格式是"ND"，自动切分开启
```

### sample1
```python
import numpy as np
tensor_np = np.array([256, 512], dtype="float32")
tensor = Tensor(tensor_np) #创建一个Tensor，存放在Global Memory上，数据类型是"FP32"，shape是[512]，存储格式是"ND"，自动切分关闭。
    tensor_1 = Tensor(tensor_np, format="NZ", multi_core=False) #创建一个Tensor，存放在Global Memory上，数据类型是"FP32"，shape是[512]，存储格式是"NZ"，自动切分开启。
```
## shape

### 函数说明  
Tensor属性，返回Tensor的shape信息。

### 参数说明  
- 无参数

### 返回值
- shape:类型为List(int)

### sample0  
```python
tensor_np = np.array([256, 512], dtype="float32")
shape = tensor_np.shape
```

## dtype

### 函数说明  
Tensor属性，返回Tensor的数据类型。

### 参数说明  
- 无参数

### 返回值
- dtype:类型为str

### sample0  
```python
tensor_np = np.array([256, 512], dtype="float32")
dtype = tensor_np.dtype
```

## mem_type

### 函数说明  
Tensor属性，返回Tensor的存储位置。

### 参数说明  
- 无参数

### 返回值
- mem_type:类型为str

### sample0  
```python
tensor_np = np.array([256, 512], dtype="float32")
mem_type = tensor_np.mem_type
```

## load(self, tensor)

### 函数说明  
Tensor方法，将另外一个Tensor对应的数据加载到当前Tensor中。

### 参数说明  
- tensor：被加载的Tensor。

### 返回值
- 无

### sample0  
```python
tensor_gm.load(tensor_ub)
```

## sync_device_to_host()

### 函数说明  
Tensor方法，Tensor对应的device数据拷贝到host上。注意，调用此函数时，确保调用位置在SWFT Kernel之外，且Tensor的存放位置在全局内存上（`mem_type="GM"`）

### 参数说明  
- 无
### 返回值
- 无

### sample0  
```python
tensor_gm.sync_device_to_host()
```


## sync_host_to_device()

### 函数说明  
Tensor方法，Tensor对应的host数据拷贝到device上。注意，调用此函数时，确保调用位置在SWFT Kernel之外，且Tensor的存放位置在全局内存上（`mem_type="GM"`）

### 参数说明  
- 无
### 返回值
- 无

### sample0  
```python
tensor_gm.sync_host_to_device()
```

## as_numpy()

### 函数说明  
Tensor方法，将Tensor存储的host数据转换为numpy.ndarray。注意，调用此函数时，确保调用位置在SWFT Kernel之外，且Tensor的存放位置在全局内存上（`mem_type="GM"`）

### 参数说明  
- 无
### 返回值
- 无

### sample0  
```python
tensor_gm.as_numpy()