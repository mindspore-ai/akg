# Benchmark

## 概述

本 benchmark 基于AI Kernel 生成评测集 [KernelBench](https://github.com/ScalingIntelligence/KernelBench)，我们在此基础上进行了前端框架的迁移和结构的合理化调整。

在`kernelbench`目录下，存放了我们为 KernelBench 任务适配的 MindSpore 和 NumPy 实现。原始的 PyTorch 实现已通过 git submodule 方式引入（位于`../thirdparty/KernelBench`），从而实现了对不同计算框架的广泛支持和验证。

在`aikgbench`目录下，提供aikg项目中有较高使用价值的Kernel场景，整理成对应的benchmark：

- **vLLM**: 基于 [vLLM](https://github.com/vllm-project/vllm) 推理引擎的高性能算子，包括注意力机制、归一化层、位置编码等核心组件
- **SGLang**: 基于 [SGLang](https://github.com/ShanHongNan/SGlang) 的高效推理算子，专注于优化的注意力kernel和状态管理