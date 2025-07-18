# SWFT

## SWFT 简介
SWFT（Seamless Wide-Range Fusion Tool）是一款面向华为昇腾芯片的算子编译器，具有昇腾亲和，极简编写等特征, 支持大范围融合算子的自动生成。SWFT旨在帮助开发者快速实现面向大模型性能优化所需的各类融合算子，降低算子开发门槛，提升开发效率。

## 功能特点
MindSpore原生：支持MindSpore侧完成开发算子、验证精度、测试性能等流程，无缝接入MindSpore生态
Ascend亲和：AscendC亲和的接口设计，方便AscendC开发者快速上手
大范围融合支持：支持大范围自定义融合算子的生成
灵活DSL设计：支持Python前端便捷表达切分方式，数据搬运逻辑，计算逻辑等，最大化硬件利用率。

## 目录结构
```shell
swft/
├── python/                     # 核心生成引擎
├────swft/
│      ├── utils/               # 合法性检查接口
│      ├── intrinsic/           # 具体指令类实现
│      ├── core/                # 核心编译模块
│      └── api/                 # dsl接口
├── lib/                        # 后端codegen
├── docs/                       # 文档目录
│   ├── slicedata.md            # 数据切分api文档
│   ├── move.md                 # 数据搬运api文档
│   ├── compute.md              # 计算api文档
│   └── composite.md            # 复合算子api文档
├── pybind/                     # Pybind相关
├── tests/                      # 测试用例
├── pyproject.toml              # 打包相关配置
├── setup.cfg                   # 打包相关配置
├── build.sh                    # 构建脚本
└── README.md                   # 项目说明文档

```
相关API文档链接：

[进阶教程.md](docs/进阶教程.md) SWFT的算子编写教程

[tensor.md](docs/tensor.md) SWFT Tensor API教程

[core.md](docs/core.md) swft.core模块API教程

[slicedata.md](docs/slicedata.md) SWFT数据切分模块API教程

[move.md](docs/move.md) SWFT数据搬运模块API教程

[compute.md](docs/compute.md) SWFT数据计算模块API教程

[composite.md](docs/composite.md) SWFT复合计算模块API教程

## 快速开始
### 环境要求
1. gcc>=9.4
2. python>=3.7
3. pybind11
4. setuptools>=42
5. build>=1.2.2.post1
6. wheel>=0.45.1

### 安装方法
自动打包和安装swft软件包：
```shell
bash build.sh
```
### 使用示例
#### 使用示例1：算子源到源编译（python -> cce）
1. 首先通过SWFT源到源生成昇腾算子源码文件，以swft\op_test\math\tanh.py为例：
首先加载SWFT API，并且编写tanh算子
```python
from swft.core import *
from swft.api import *

@sub_kernel(core_num=8)
def tanh_kernel(x, out):
    x_ub = move_to_ub(x)
    tanh_ub = tanh(x_ub)
    out.load(tanh_ub)
```
其中`@sub_kernel`指示了当前函数需要通过SWFT进行算子编译。`core_num`指示了编译该算子需要用到的昇腾AICore核数，`tanh_kernel`内部使用的API：`move_to_ub`，`tanh`，`Tensor.load`等，是SWFT对外提供的昇腾亲和的函数式API（详情请参阅[slicedata.md](docs/slicedata.md)，[move.md](docs/move.md)，[compute.md](docs/compute.md)，[composite.md](docs/composite.md), [tensor.md](docs/tensor.md)）

2. SWFT算子定义完成后，通过如下代码启动算子编译，并最终输出编译后的算子源码。
```python
set_context("310P") #指示编译的昇腾后端，当前仅支持310系列
x = Tensor("GM", "FP16", [512], format="ND", multi_core=True)
out = Tensor("GM", "FP16", [512], format="ND", multi_core=True)
tanh_kernel(x, out)
compile_kernel(f"./temp/tanh_kernel/tanh_kernel.cce", "tanh_kernel") #指示算子编译输出文件的最终位置，输出为CCE代码。
```

3. 我们提供了测试代码的自动编译生成功能，可以一键生成算子驱动代码，编译链接并执行算子：
```python
exec_kernel("tanh_kernel", locals(), prefix_path="temp", inputs=['x'], outputs=['out'])
```
完整示例请参考：[tanh.py](op_test/math/tanh.py)

我们同时提供了更多的算子用例，请参考：`op_test/` 目录

#### 使用示例2：MindSpore原生调用SWFT算子
TODO

#### 使用示例3：Python原生的算子即时编译与执行：
我们同时提供了Python原生的SWFT执行方式，仍然使用tanh作为示例：
首先加载SWFT API，并且编写tanh算子,使用@jit进行函数装饰，表示在python执行时进行及时算子编译（Just-In-Time compilation）。

```python
import numpy as nps
from swft.core import *
from swft.api import *
from swft.runtime import * # swft 昇腾执行环境

@jit(core_num=8)
def tanh_kernel(x, out):
    x_ub = move_to_ub(x)
    tanh_ub = tanh(x_ub)
    out.load(tanh_ub)
```

2. SWFT算子定义完成后，通过如下代码启动算子编译，加载编译后的算子二进制进入python环境，调用算子，并最终输出算子执行后的结果。
```python
npu_session = NPUSession.create(device_id=0, context="310P") # 使用0号NPU进行算子执行，并指定编译的昇腾后端，当前仅支持310系列
x_np = np.random.uniform(-1, 1, [512]).astype(np.float16)
x = Tensor(x_np, multi_core=True) # 直接从numpy array初始化SWFT Tensor
out = Tensor("GM", "FP16", [512], format="ND", multi_core=True)
tanh_kernel(x, out) # 编译算子，加载算子，执行算子
npu_session.sync_stream() # 同步device流
out.sync_device_to_host() # 将结果从device侧拷贝回host侧
print(out.as_numpy()) # 将结果以numpy.ndarray形式导出
```
## 致谢
感谢以下工具对本工具的前端API设计的启发和支持：
    MindStudio Kernel Performance Prediction (MsKPP)
    https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/devaids/optool/atlasopdev_16_0006.html
