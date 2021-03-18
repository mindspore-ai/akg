- [What Is AKG?](#what-is-akg)
- [Hardware Backends Support](#hardware-backends-support)
- [Build](#build)
    - [Build With MindSpore](#build-with-mindspore)
    - [Build Standalone](#build-standalone)
- [Run](#run)
- [Contributing](#contributing)
- [Release Notes](#release-notes)
- [License](#license)

[查看中文](./README_CN.md)

## What Is AKG
AKG(Auto Kernel Generator) is an optimizer for operators in Deep Learning Networks. It provides the ability to automatically fuse ops with specific patterns. AKG works with MindSpore-GraphKernel to improve the performance of networks running on different hardware backends.

AKG composes with three basic optimization module, normalization, auto schedule and backend optimization.
- **normalization.** In order to solve the limitation in expression ability of polyhedral(which can only process static linear programs), the computation IR needs to be normalized first. The mainly optimization of normalization module includes auto-inline, loop fusing, common subexpression elimination and so on.
- **auto schedule.** Base on polyhedral technology, the auto schedule module mainly have auto-vectorization, auto-tiling, thread/block mapping, dependency analysis and memory promotion.
- **backend optimization.** The backend optimization module mainly consists of TensorCore acceleration, double buffer optimization, storage flatten optimization and inject sync optimization.

  <img src="docs/akg-design.png" style="zoom:80%" div align=center/>

## Hardware Backends Support
At present, `GPU V100/A100` are supported. More Backends are on the list.

## Build

### Build With MindSpore
See [MindSpore README.md](https://gitee.com/mindspore/mindspore/blob/master/README.md) for details.

### Build Standalone
We suggest you build and run akg together with MindSpore. And we also provide a way to run case in standalone mode for convenience sake.
Refer to [MindSpore Installation](https://www.mindspore.cn/install/en) for more information about compilation dependencies.
  ```
  bash build.sh -e $target // target can set 'gpu'
  ```

## Run Standalone
1. Set Environment

- GPU V100/A100
  ```
  cd tests
  source ./test_env.sh gpu
  ```

2. Run test

- GPU V100/A100
  ```
  cd tests/operators/gpu
  python3 test_all.py -s "op_name" #replace op_name with the operator name which you want to test
  ```

## Contributing

Welcome contributions. See [MindSpore Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for
more details.

## Release Notes

The release notes, see our [RELEASE](RELEASE.md).

## License

[Apache License 2.0](LICENSE)
