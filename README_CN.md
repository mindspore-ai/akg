- [AKG简述](#AKG简述)
- [硬件后端支持](#硬件后端支持)
- [构建](#构建)
    - [从MindSpore侧构建](#从MindSpore侧构建)
    - [独立构建](#独立构建)
- [运行](#运行)
- [贡献](#贡献)
- [版本说明](#版本说明)
- [许可证](#许可证)

[View English](./README.md)

## AKG简述
AKG(Auto Kernel Generator)对深度神经网络中的算子进行优化，并提供特定模式下的算子自动融合功能。AKG与MindSpore的图算融合功能协同工作，可提升在不同硬件后端上运行网络的性能。

AKG由四个基本的优化模块组成：规范化、自动调度、指令发射和后端优化。
- **规范化：** 为了解决polyhedral表达能力的局限性（只能处理静态的线性程序），需要首先对计算公式IR进行规范化。规范化模块中的优化主要包括自动运算符inline、循环拆分和公共子表达式优化等。
- **自动调度：** 自动调度模块基于polyhedral技术，主要包括自动向量化、自动切分、依赖分析和数据搬移等。
- **指令发射：** 指令发射模块的优化主要包括循环规范化、标签自动生成和指令发射等。
- **后端优化：** 后端优化模块的优化主要包括双缓冲区、存储重写和同步指令插入等。

  <img src="docs/akg-design.png" style="zoom:80%" div align=center/>

## 硬件后端支持
当前仅支持`Ascend910`，更多硬件后端支持待开发。

## 构建

### 从MindSpore侧构建
详细细节请参考[MindSpore README.md](https://gitee.com/mindspore/mindspore/blob/master/README.md)。

### 独立构建
我们建议您从MindSpore侧构建运行AKG代码，但同时为了方便开发，我们提供了独立编译运行AKG的方式。
独立构建模式下需要Ascend平台的支持，详细的编译依赖请参考[MindSpore安装指南](https://www.mindspore.cn/install)。
  ```
  bash build.sh
  ```
## 运行
1. 设置环境变量
  ```
  cd tests
  source ./test_env.sh amd64
  export RUNTIME_MODE='air_cloud'
  export PATH=${PATH}:${YOUR_CCEC_COMPILER_PATH}
  ```

2. 运行测试用例
  ```
  cd tests/operators/vector
  pytest -s test_abs_001.py -m "level0" # 运行level0测试用例
  ```

## 贡献

欢迎您的贡献，具体细节请参考[MindSpore贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 版本说明

版本说明详见[RELEASE](RELEASE.md).

## 许可证

[Apache License 2.0](LICENSE)
