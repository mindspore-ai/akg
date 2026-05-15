# thirdparty

本目录用于存放按需下载的第三方 benchmark 仓库。

统一通过项目根目录的下载脚本管理：

```bash
bash download.sh --with_kernelbench
bash download.sh --with_multikernelbench
bash download.sh --with_evokernel
bash download.sh --with_sol_execbench

# 一次性下载全部 benchmark
bash download.sh --with_all_benchmarks
```

这些仓库会被 clone 到当前目录，并自动 checkout 到项目固定的 commit。
