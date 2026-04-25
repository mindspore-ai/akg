#!/bin/bash
set -e

# 解析命令行参数
WITH_LOCAL_MODEL=false
WITH_SOL_EXECBENCH=false
WITH_KERNELBENCH=false
WITH_MULTIKERNELBENCH=false
WITH_EVOKERNEL=false
WITH_SOLAR=false
WITH_ALL_BENCHMARKS=false
for arg in "$@"; do
  if [ "$arg" = "--with_local_model" ]; then
    WITH_LOCAL_MODEL=true
  elif [ "$arg" = "--with_sol_execbench" ]; then
    WITH_SOL_EXECBENCH=true
  elif [ "$arg" = "--with_kernelbench" ]; then
    WITH_KERNELBENCH=true
  elif [ "$arg" = "--with_multikernelbench" ]; then
    WITH_MULTIKERNELBENCH=true
  elif [ "$arg" = "--with_evokernel" ]; then
    WITH_EVOKERNEL=true
  elif [ "$arg" = "--with_solar" ]; then
    WITH_SOLAR=true
  elif [ "$arg" = "--with_all_benchmarks" ]; then
    WITH_ALL_BENCHMARKS=true
  fi
done

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型目标路径
MODEL_DIR="$HOME/.akg_agents/text2vec-large-chinese"
THIRDPARTY_DIR="${PROJECT_ROOT}/thirdparty"
SOL_EXECBENCH_DIR="${THIRDPARTY_DIR}/sol-execbench"
SOL_EXECBENCH_FALLBACK_DIR="${THIRDPARTY_DIR}/SOL-ExecBench-dataset"
KERNELBENCH_DIR="${THIRDPARTY_DIR}/KernelBench"
MULTIKERNELBENCH_DIR="${THIRDPARTY_DIR}/MultiKernelBench"
EVOKERNEL_DIR="${THIRDPARTY_DIR}/EvoKernel"
SOLAR_DIR="${SOLAR_DIR:-${THIRDPARTY_DIR}/SOLAR}"

KERNELBENCH_REPO_URL="https://github.com/ScalingIntelligence/KernelBench.git"
# 优先以历史 .gitmodules 中记录的 commit 为准。
KERNELBENCH_COMMIT="21fbe5a642898cd60b8f60c7aefb43d475e11f33"
MULTIKERNELBENCH_REPO_URL="https://github.com/wzzll123/MultiKernelBench.git"
MULTIKERNELBENCH_COMMIT="55cb5c059573f0bf00f2dc24c75f810059cf2785"
EVOKERNEL_REPO_URL="https://huggingface.co/datasets/noahli/EvoKernel"
EVOKERNEL_COMMIT="af61b2a307d6d9f8d313893f2c87414c51a97863"
SOLAR_REPO_URL="${SOLAR_REPO_URL:-https://github.com/NVlabs/SOLAR.git}"
SOLAR_REF="${SOLAR_REF:-}"

function check_python_and_deps() {
  if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 python3，请安装 Python 3.10/3.11/3.12"
    exit 1
  fi

  if ! command -v pip3 &> /dev/null; then
    echo "错误：未找到 pip3，请安装 pip3"
    exit 1
  fi
}

function check_git() {
  if ! command -v git &> /dev/null; then
    echo "错误：未找到 git，请先安装 git"
    exit 1
  fi
}

function download_text2vec_large_chinese_lib() {
  ensure_python_modules "huggingface_hub"
  mkdir -p "$HOME/.akg_agents"

  if [ -d "$MODEL_DIR" ]; then
    echo "模型目录已存在：$MODEL_DIR，跳过下载。如需重新下载，请先删除该目录。"
    return 0
  fi

  echo "正在下载 text2vec-large-chinese 模型..."
  
  # 使用 Python 脚本下载
  python3 -c "
import os, sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id='GanymedeNil/text2vec-large-chinese',
        local_dir='$MODEL_DIR',
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print('✅ 模型下载成功！路径：$MODEL_DIR')
except Exception as e:
    print(f'❌ 下载失败: {e}', file=sys.stderr)
    sys.exit(1)
  "
}

function ensure_python_modules() {
  local missing_modules=()

  for module in "$@"; do
    if ! python3 -c "import ${module}" &> /dev/null; then
      missing_modules+=("${module}")
    fi
  done

  if [ ${#missing_modules[@]} -eq 0 ]; then
    return 0
  fi

  echo "检测到缺少 Python 依赖：${missing_modules[*]}，正在尝试安装..."
  if ! pip3 install --user "${missing_modules[@]}"; then
    echo "错误：无法安装依赖，请手动运行：pip3 install --user ${missing_modules[*]}"
    exit 1
  fi
}

function clone_and_checkout_repo() {
  local repo_name="$1"
  local repo_url="$2"
  local target_dir="$3"
  local target_commit="$4"
  local pull_lfs="${5:-false}"

  mkdir -p "$(dirname "${target_dir}")"

  if [ -d "${target_dir}/.git" ]; then
    echo "${repo_name} 已存在，更新远端信息..."
    git -C "${target_dir}" fetch --tags origin
  else
    if [ -d "${target_dir}" ] && [ -n "$(ls -A "${target_dir}" 2>/dev/null)" ]; then
      echo "错误：目录 ${target_dir} 已存在且非空，但不是 git 仓库，请手动清理后重试。"
      exit 1
    fi

    echo "克隆 ${repo_name} 到 ${target_dir}..."
    git clone "${repo_url}" "${target_dir}"
  fi

  if [ -n "${target_commit}" ]; then
    if ! git -C "${target_dir}" rev-parse --verify "${target_commit}^{commit}" >/dev/null 2>&1; then
      echo "尝试拉取 ${repo_name} 的目标 commit: ${target_commit}"
      git -C "${target_dir}" fetch origin "${target_commit}" || true
    fi

    if git -C "${target_dir}" rev-parse --verify "${target_commit}^{commit}" >/dev/null 2>&1; then
      echo "切换 ${repo_name} 到 commit ${target_commit}..."
      git -C "${target_dir}" checkout "${target_commit}"
    else
      echo "警告：未找到 ${repo_name} 的目标 commit ${target_commit}，保留当前分支状态。"
    fi
  fi

  if [ "${pull_lfs}" = "true" ]; then
    if command -v git-lfs &> /dev/null; then
      echo "检测到 git-lfs，拉取 ${repo_name} 的 LFS 文件..."
      git -C "${target_dir}" lfs pull
    else
      echo "警告：未检测到 git-lfs，${repo_name} 中如果包含 LFS 文件，可能只会拿到指针文件。"
    fi
  fi
}

function download_sol_execbench() {
  ensure_python_modules "datasets" "huggingface_hub"

  echo "开始下载 SOL-ExecBench 数据集..."
  mkdir -p "${THIRDPARTY_DIR}"

  if [ ! -d "${SOL_EXECBENCH_DIR}" ]; then
    echo "克隆 nvidia/sol-execbench 仓库到 thirdparty/sol-execbench..."
    git clone https://github.com/nvidia/sol-execbench "${SOL_EXECBENCH_DIR}"
  else
    echo "仓库 sol-execbench 已存在，跳过克隆。"
  fi

  cd "${SOL_EXECBENCH_DIR}"

  echo "运行下载脚本从 HuggingFace 拉取数据..."
  set +e
  python3 scripts/download_solexecbench.py
  DOWNLOAD_STATUS=$?
  set -e

  if [ "${DOWNLOAD_STATUS}" -ne 0 ]; then
    echo "HuggingFace 下载失败，尝试从 GitCode 备用仓库下载数据集..."

    if [ ! -d "${SOL_EXECBENCH_FALLBACK_DIR}" ]; then
      git clone https://gitcode.com/yiyanzhi_akane1/SOL-ExecBench.git "${SOL_EXECBENCH_FALLBACK_DIR}"
    else
      echo "备用仓库 SOL-ExecBench-dataset 已存在，跳过克隆。"
    fi

    python3 -c "
from pathlib import Path
path = Path('scripts/download_solexecbench.py')
content = path.read_text(encoding='utf-8')
updated = content.replace('\"nvidia/SOL-ExecBench\"', '\"../SOL-ExecBench-dataset\"')
path.write_text(updated, encoding='utf-8')
"

    echo "运行下载脚本从本地备用仓库拉取数据..."
    python3 scripts/download_solexecbench.py
  fi

  echo "下载完成！数据存放在 thirdparty/sol-execbench/data/benchmark/"
}

function download_kernelbench() {
  check_git
  clone_and_checkout_repo "KernelBench" "${KERNELBENCH_REPO_URL}" "${KERNELBENCH_DIR}" "${KERNELBENCH_COMMIT}"
}

function download_multikernelbench() {
  check_git
  clone_and_checkout_repo "MultiKernelBench" "${MULTIKERNELBENCH_REPO_URL}" "${MULTIKERNELBENCH_DIR}" "${MULTIKERNELBENCH_COMMIT}"
}

function download_evokernel() {
  check_git
  clone_and_checkout_repo "EvoKernel" "${EVOKERNEL_REPO_URL}" "${EVOKERNEL_DIR}" "${EVOKERNEL_COMMIT}" "true"
}

function download_and_install_solar() {
  check_git

  echo "开始下载并安装 Solar..."
  clone_and_checkout_repo "Solar" "${SOLAR_REPO_URL}" "${SOLAR_DIR}" "${SOLAR_REF}"

  if [ -f "${SOLAR_DIR}/install.sh" ]; then
    echo "使用 Solar 自带 install.sh 安装依赖..."
    bash "${SOLAR_DIR}/install.sh" --skip-torch
  elif [ -f "${SOLAR_DIR}/requirements.txt" ]; then
    echo "未找到 Solar install.sh，回退为 requirements.txt 安装..."
    pip3 install -r "${SOLAR_DIR}/requirements.txt"
  else
    echo "错误：Solar 仓库缺少 install.sh / requirements.txt: ${SOLAR_DIR}"
    exit 1
  fi

  if [ -f "${SOLAR_DIR}/setup.py" ] || [ -f "${SOLAR_DIR}/pyproject.toml" ]; then
    echo "安装 Solar Python 包..."
    pip3 install -e "${SOLAR_DIR}" --no-deps
  else
    echo "错误：Solar 仓库缺少 setup.py / pyproject.toml: ${SOLAR_DIR}"
    exit 1
  fi

  echo "校验 Solar 安装..."
  python3 - <<'PY'
import solar
from solar.graph import PyTorchProcessor
from solar.einsum import PyTorchToEinsum
from solar.analysis import EinsumGraphAnalyzer
from solar.perf import EinsumGraphPerfModel
print("solar import ok:", solar.__file__)
print("solar api ok")
PY
}

# 主逻辑
if $WITH_ALL_BENCHMARKS; then
  WITH_SOL_EXECBENCH=true
  WITH_KERNELBENCH=true
  WITH_MULTIKERNELBENCH=true
  WITH_EVOKERNEL=true
fi

if $WITH_LOCAL_MODEL; then
  check_python_and_deps
  download_text2vec_large_chinese_lib
fi

if $WITH_SOL_EXECBENCH; then
  check_python_and_deps
  download_sol_execbench
fi

if $WITH_KERNELBENCH; then
  download_kernelbench
fi

if $WITH_MULTIKERNELBENCH; then
  download_multikernelbench
fi

if $WITH_EVOKERNEL; then
  download_evokernel
fi

if $WITH_SOLAR; then
  check_python_and_deps
  download_and_install_solar
fi
