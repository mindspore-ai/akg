## 环境配置
- HOME_DIR: /home/user
- ENV_TYPE: conda
- CONDA_ENV: akg_agents
- AKG_AGENTS_DIR: /home/user/akg/akg_agents

## 命令模板
将 <CMD> 替换为实际命令：
conda run -n akg_agents --no-capture-output bash -c "cd /home/user/akg/akg_agents && source env.sh && <CMD>"

## 验证状态
- akg_cli: ✅
- LLM: ✅

## 硬件
- CPU: x86_64 (Intel Xeon Platinum 8350C)
- GPU: NVIDIA A100-SXM4-80GB × 8
- 内存: 1007Gi

## Framework
- torch: 2.1.0 (CUDA: True)
- torch_npu: 未安装

## DSL
- triton: 2.1.0
- g++/clang++: g++ (GCC) 10.2.1

## 可用后端 & DSL & 架构
| 后端 | DSL | 架构 |
|------|-----|------|
| cuda | triton_cuda | a100 |
| cpu | cpp | x86_64 |
