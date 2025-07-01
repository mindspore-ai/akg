# Benchmark

## 概述

基于[KernelBench](https://github.com/ScalingIntelligence/KernelBench)与AIKG自定义任务集合，构建一个专注于AI Kernel 生成的Benchmark。

在`kernelbench`目录下，保留原始KernelBench中的PyTorch实现同时为每个任务提供了相关的MindSpore、NumPy任务，以支持更广泛的计算框架验证和性能对比。

在`aikg_bench`目录下，提供aikg项目中有较高使用价值的Kernel场景，整理成对应的benchmark。