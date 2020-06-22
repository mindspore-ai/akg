- [What Is AKG?](#what-is-akg)
- [Hardware Backends Support](#hardware-backends-support)
- [Build](#build)
    - [Build With MindSpore](#build-with-mindspore)
    - [Build Standalone](#build-standalone)
- [Run](#run)
- [Contributing](#contributing)
- [Release Notes](#release-notes)
- [License](#license)

## What Is AKG
AKG(Auto Kernel Generator) is an optimizer for operators in Deep Learning Networks. It provides the ability to automatically fuse ops with specific patterns. AKG works with MindSpore-GraphKernel to improve the performance of networks running on different hardware backends.

AKG composes with four basic optimization module, normalization, auto schedule, instruction emit and backend optimization.
- **normalization.** The mainly optimization of normalization includes three address transform, common subexpression elimination, copy propagation and so on.
- **auto schedule.** The auto schedule module mainly have vectorization, loop tiling, mem promotion and loop distribution.
- **instruction emit.** The instruction emitting module has the optimization about loop normalization, auto pragma and emit instruction.
- **backend optimization.** The backend optimization module consists of double buffer optimization, storage rewrite optimization and inject sync optimization.

  <img src="docs/akg-design.png" style="zoom:80%" div align=center/>

## Hardware Backends Support
At present, `Ascend910` is supported only. More Backends are on the list.

## Build

### Build With MindSpore
See [MindSpore README.md](https://gitee.com/mindspore/mindspore/blob/master/README.md) for details.

### Build Standalone
We suggest you build and run akg together with MindSpore. And we also provide a way to run case in standalone mode for convenience sake.
Ascend platform is needed to build this mode. Refer to [MindSpore Installation](https://www.mindspore.cn/install/en) for more information about compilation dependencies.
  ```
  git submodule update --init
  bash build.sh
  ```
## Run Standalone
1. Set Environment
  ```
  cd tests
  source ./test_env.sh amd64
  export RUNTIME_MODE='air_cloud'
  export PATH=${PATH}:${YOUR_CCEC_COMPILER_PATH}
  ```

2. Run test
  ```
  cd tests/operators/vector
  pytest -s test_abs_001.py -m "level0" # run level0 testcases
  ```

## Contributing

Welcome contributions. See [MindSpore Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for
more details.

## Release Notes

The release notes, see our [RELEASE](RELEASE.md).

## License

[Apache License 2.0](LICENSE)
