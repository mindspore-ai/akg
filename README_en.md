# Auto Kernel Generator (AKG)

#### Project Introduction

AKG (Auto Kernel Generator) is a deep learning compiler designed to optimize operators in deep neural networks and provide automatic operator fusion capabilities under specific patterns.  
After version 2.4, we have completely upgraded AKG. The current AKG project includes two subprojects: AIKG and AKG-MLIR. It supports automatic operator generation for multiple hardware backends such as CPU, NVIDIA V100/A100, Atlas 800T A2/A3, and 300I DUO.

- **AIKG:** AIKG (AI Driven Kernel Generator) is our latest incubated tool for AI-driven operator generation. It provides a Multi-agent-based operator auto-generation framework, the AscendKernelBench evaluation platform, and an operator knowledge base. Currently, it supports generating Triton-Ascend (Atlas A2/A3) and SWFT (300I DUO) DSLs through LLM and compiling operator kernels.
- **AKG-MLIR:** A deep learning compiler evolved from the open-source MLIR project, providing a complete operator compilation pipeline for CPU/GPU/Ascend. It currently includes the MindSpore Dialect for graph compilation and extends dialects such as Linalg, Affine, and GPU, enhancing loop fusion scheduling capabilities. Additionally, AKG-MLIR integrates with AscendNPU IR, supporting fused operator generation for the Ascend backend.

#### Installation and Usage Instructions

Each subproject in the AKG repository supports independent installation and building. Please refer to the respective documentation for each subproject.

- **AIKG:** Refer to the [AIKG](./aikg/README.md) documentation.
- **AKG-MLIR:** Refer to the [AKG-MLIR](./akg-mlir/README.md) documentation.

## Contributions

We welcome your contributions. For details, please refer to the [MindSpore Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md).

## Release Notes

For release notes, see [RELEASE](RELEASE.md).

## License
[Apache License 2.0](LICENSE).

## AKG SIG WeChat Group

<div>  
  <a href="docs/AKG_QRCode.png">
    <img src="docs/AKG_QRCode.png" alt="AKG SIG WeChat Group QR Code" width="220">  
  </a>
</div>
