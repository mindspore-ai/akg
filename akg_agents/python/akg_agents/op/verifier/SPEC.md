# op/verifier/ — 验证器

## 职责

验证生成的内核代码的正确性和性能。通过三类适配器（backend、dsl、framework）组合支持多平台。

## 目录结构

```
verifier/
├── kernel_verifier.py         # KernelVerifier — 验证入口，组合三类适配器
├── sol_verifier.py            # SOL-ExecBench 格式验证的专用生成器
├── profiler.py                # NPU/CUDA 性能采集
├── profiler_utils.py          # profile 脚本执行、msprof/nsys 解析
├── l2_cache_clear.py          # Ascend L2 cache 清理
└── adapters/                  # 三类适配器
    ├── factory.py             # get_backend_adapter / get_dsl_adapter / get_framework_adapter
    ├── backend/               # BackendAdapter 及实现
    │   ├── base.py            #   BackendAdapter(ABC)
    │   ├── cuda.py            #   BackendAdapterCuda
    │   ├── ascend.py          #   BackendAdapterAscend
    │   └── cpu.py             #   BackendAdapterCpu
    ├── dsl/                   # DSLAdapter 及实现
    │   ├── base.py            #   DSLAdapter(ABC)
    │   ├── triton_cuda.py     #   DSLAdapterTritonCuda
    │   ├── triton_ascend.py   #   DSLAdapterTritonAscend
    │   ├── cpp.py             #   DSLAdapterCpp
    │   ├── cuda_c.py          #   DSLAdapterCudaC
    │   ├── ascendc.py         #   DSLAdapterAscendC
    │   ├── tilelang_cuda.py   #   DSLAdapterTilelangCuda
    │   ├── tilelang_npuir.py  #   DSLAdapterTilelangNpuir
    │   ├── torch.py           #   DSLAdapterTorch
    │   ├── pypto.py           #   DSLAdapterPypto
    │   └── swft.py            #   DSLAdapterSwft
    └── framework/             # FrameworkAdapter 及实现
        ├── base.py            #   FrameworkAdapter(ABC)
        ├── torch.py           #   FrameworkAdapterTorch
        ├── mindspore.py       #   FrameworkAdapterMindSpore
        └── numpy.py           #   FrameworkAdapterNumpy
```

## 开发约定

### 新增验证适配器的标准流程

1. 确定适配器类型（backend / dsl / framework）
2. 在对应子目录创建文件，继承 `BackendAdapter(ABC)` / `DSLAdapter(ABC)` / `FrameworkAdapter(ABC)`
3. 实现所有抽象方法
4. 在 `adapters/factory.py` 中注册名称映射

### KernelVerifier 核心逻辑

`KernelVerifier` 通过 `get_*_adapter` 工厂方法获取三个适配器实例，然后组合生成验证脚本（Jinja2 模板）、CMake 配置等，最终执行验证和 profiling。

## 不做什么

- **不要**在适配器中实现 Agent/Workflow 逻辑
- **不要**硬编码后端/DSL 特定行为到 `kernel_verifier.py`——通过适配器扩展
