# Auto Kernel Generator (AKG)

#### 项目简介

AKG(Auto Kernel Generator)是一个深度学习编译器，用以对深度神经网络中的算子进行优化，并提供特定模式下的算子自动融合功能。
从 2.4 版本之后，我们对 AKG 进行了全新升级，当前的 AKG 项目中，包含了 AIKG、AKG-MLIR 2个子项目，支持 CPU、NVIDIA、V100/A100、Atlas 800T A2/A3、300I DUO 等多硬件后端的算子自动生成。

- **AIKG：** AIKG(AI Driven Kernel Generator)是我们最新孵化的一款大模型驱动的算子生成工具，提供了基于 Multi-agent 的算子自动生成框架、AscendKernelBench 评测平台、以及算子知识库。当前支持通过 LLM 的方式，生成 Triton-Ascend（Atalas A2/A3）以及 SWFT（300I DUO）的 DSL，并完成算子 kernel 编译。
- **AKG-MLIR：** 基于 MLIR 开源项目演进的深度学习编译器，提供了 CPU/GPU/Ascend 上完整的算子编译 Pipeline。当前包含了 MindSpore Dialect 图编译方言，并对 Linalg、Affine、GPU 等方言进行了扩展，增强了循环融合调度能力。此外，AKG-MLIR 对接了 AscendNPU IR，支持昇腾后端融合算子生成。

#### 安装及使用说明

AKG 代码仓中的各个子项目，都支持独立安装构建，请参考各子项目使用说明。

- **AIKG：** 请参考 [AIKG](./aikg/README_CN.md) 文档
- **AKG-MLIR：** 请参考 [AKG-MLIR](./akg-mlir/README.md) 文档

## 贡献

欢迎您的贡献，具体细节请参考[MindSpore 贡献者 Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 版本说明

版本说明详见[RELEASE](RELEASE.md)。

## 许可证
[Apache License 2.0](LICENSE)。

## AKG SIG 交流微信群

<div>
  <a href="docs/AKG_QRCode.png">
    <img src="docs/AKG_QRCode.png" alt="AKG SIG 交流微信群二维码" width="220">
  </a>
</div>
