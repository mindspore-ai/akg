# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

#!/bin/bash

# ==============================================================================
# AscendC Memcheck 自动化脚本 - 执行3 步（编译、安装、memcheck）
# ==============================================================================
#
# 功能：
#   1. 编译算子（带 sanitizer 选项）
#   2. 安装算子包
#   3. 运行 memcheck 检测
#
# 用法：
#   ./scripts/run_memcheck_pre.sh [options]
#
# 参数：
#   -h, --help          显示帮助信息
#   -c, --config FILE   配置文件路径（默认：./memcheck_input.json）
#   --skip-build        跳过编译步骤
#   --keep-build        保留构建目录
#   --verbose           显示详细输出
#
# ==============================================================================

set -e

# ============================================================================
# 颜色输出定义
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 日志函数
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# ============================================================================
# 默认配置
# ============================================================================

CONFIG_FILE="./memcheck_input.json"
SKIP_BUILD=false
KEEP_BUILD=false
VERBOSE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# 显示帮助信息
# ============================================================================

show_help() {
    cat << EOF
AscendC Memcheck 自动化脚本 - 执行第 1-3 步（编译、安装、memcheck）

用法：
  $(basename "$0") [options]

参数：
  -h, --help          显示帮助信息
  -c, --config FILE   配置文件路径（默认：./memcheck_input.json）
  -o, --output DIR    输出目录（默认：./memcheck_output）
  --skip-build        跳过编译步骤
  --keep-build        保留构建目录
  --verbose           显示详细输出

示例：
  # 使用默认配置
  $(basename "$0")

  # 指定配置文件
  $(basename "$0") --config /path/to/memcheck_input.json

  # 跳过编译步骤
  $(basename "$0") --skip-build

  # 显示详细输出
  $(basename "$0") --verbose

配置文件模板：
  memcheck_input.json.template

输出目录结构：
  $OUTPUT_DIR/
  ├── status.txt              # 执行状态摘要
  ├── build/                  # 构建相关
  │   ├── build.log
  │   └── build_errors.log
  ├── install/                # 安装相关
  │   ├── install.log
  │   └── install_errors.log
  ├── memcheck/               # Memcheck 相关
  │   ├── memcheck.log
  │   ├── ascendc_memcheck_report.txt
  │   └── mindstudio_sanitizer_log/
  └── timestamp.txt           # 执行时间戳

EOF
}

# ============================================================================
# 解析命令行参数
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --keep-build)
                KEEP_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# 解析 JSON 配置文件
# ============================================================================

parse_config() {
    local config_file="$1"

    if [ ! -f "$config_file" ]; then
        log_error "配置文件不存在: $config_file"
        exit 1
    fi

    log_info "读取配置文件: $config_file"

    # 使用 Python 解析 JSON（更可靠）
    if ! command -v python3 &> /dev/null; then
        log_error "需要 Python 3 来解析配置文件"
        exit 1
    fi

    # 提取配置值
    OP_NAME=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('operator', {}).get('name', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 operator.name"; exit 1; }

    CODE_BASE_DIR=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('paths', {}).get('code_base_dir', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 paths.code_base_dir"; exit 1; }
    TEST_SCRIPT_DIR=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('testing', {}).get('test_script_dir', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 testing.test_script_dir"; exit 1; }


    TEST_SCRIPT_EXE=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('testing', {}).get('test_script_exe', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 testing.test_script_exe"; exit 1; }

    DEVICE_TYPE=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('environment', {}).get('device_type', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 environment.device_type"; exit 1; }

    CANN_ENV=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('environment', {}).get('cann_env', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 environment.cann_env"; exit 1; }
    LOAD_ENV=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    val = config.get('installation', {}).get('load_environment', True)
    print('true' if val else 'false')
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 installation.load_environment"; exit 1; }

    SANITIZER_OPTS=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('compilation', {}).get('sanitizer_options', '-sanitizer;-g'))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 compilation.sanitizer_options"; exit 1; }

    LOG_LEVEL=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('memcheck', {}).get('log_level', '3'))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 memcheck.log_level"; exit 1; }

    SLOG_PRINT=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('memcheck', {}).get('slog_print_to_stdout', 'true'))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 memcheck.slog_print_to_stdout"; exit 1; }

    TIMEOUT=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('memcheck', {}).get('timeout', 600))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 memcheck.timeout"; exit 1; }

    REBUILD=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    val = config.get('options', {}).get('rebuild', True)
    print('true' if val else 'false')
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "无法解析 options.rebuild"; exit 1; }

    # 验证必需字段
    if [ -z "$OP_NAME" ]; then
        log_error "配置文件缺少必需字段: operator.name"
        exit 1
    fi
    if [ -z "$CODE_BASE_DIR" ]; then
        log_error "配置文件缺少必需字段: paths.code_base_dir"
        exit 1
    fi
    if [ -z "$DEVICE_TYPE" ]; then
        log_error "配置文件缺少必需字段: environment.device_type"
        exit 1
    fi
    if [ -z "$CANN_ENV" ]; then
        log_error "配置文件缺少必需字段: environment.cann_env"
        exit 1
    fi
    # 构建custom安装路径（code_base_dir/custom_output）
    INSTALL_PATH="$CODE_BASE_DIR/custom_output_dir"
    log_verbose "配置解析完成:"
    log_verbose "  OP_NAME: $OP_NAME"
    log_verbose "  CODE_BASE_DIR: $CODE_BASE_DIR"
    log_verbose "  TEST_SCRIPT_EXE: $TEST_SCRIPT_EXE"
    log_verbose "  DEVICE_TYPE: $DEVICE_TYPE"
    log_verbose "  CANN_ENV: $CANN_ENV"
    log_verbose "  INSTALL_PATH: $INSTALL_PATH"
    log_verbose "  SANITIZER_OPTS: $SANITIZER_OPTS"
    log_verbose "  TEST_SCRIPT_DIR: $TEST_SCRIPT_DIR"
    log_verbose "  REBUILD: $REBUILD"
}

# ============================================================================
# 环境验证
# ============================================================================

check_environment() {
    log_info "===== 验证运行环境 ====="
    echo "CANN_ENV:$CANN_ENV"

    # 检查 CANN 环境
    if [ ! -f "$CANN_ENV/bin/setenv.bash" ]; then
        log_error "CANN 环境脚本不存在: $CANN_ENV/bin/setenv.bash"
        exit 1
    fi
    log_success "CANN 环境脚本: $CANN_ENV"

    # 检查代码目录
    if [ ! -d "$CODE_BASE_DIR" ]; then
        log_error "代码目录不存在: $CODE_BASE_DIR"
        exit 1
    fi

    BUILD_SCRIPT="$CODE_BASE_DIR/build.sh"
    if [ ! -f "$BUILD_SCRIPT" ]; then
        log_error "编译脚本不存在: $BUILD_SCRIPT"
        exit 1
    fi
    log_success "代码目录: $CODE_BASE_DIR"
    log_success "编译脚本: $BUILD_SCRIPT"

    # 检查 mssanitizer
    if ! command -v mssanitizer &> /dev/null; then
        log_warning "mssanitizer 未在 PATH 中，将在加载 CANN 环境后重试"
    else
        log_success "mssanitizer: $(command -v mssanitizer)"
    fi

    # 创建输出目录
    OUTPUT_DIR="$CODE_BASE_DIR/memcheck_output"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/build"
    mkdir -p "$OUTPUT_DIR/install"
    mkdir -p "$OUTPUT_DIR/memcheck"
    log_success "输出目录: $OUTPUT_DIR"
}

# ============================================================================
# 第 1 步：编译算子
# ============================================================================

build_operator() {
    log_info "===== 第 1 步：编译算子 ====="

    # 检查是否跳过编译
    if [ "$SKIP_BUILD" = true ] || [ "$REBUILD" = false ]; then
        log_warning "跳过编译步骤"
        return 0
    fi

    # 加载 CANN 环境
    source "$CANN_ENV/bin/setenv.bash"
    log_info "加载 CANN 环境: source $CANN_ENV/bin/setenv.bash"

    # 进入代码目录
    log_info "进入代码目录: $CODE_BASE_DIR"
    cd "$CODE_BASE_DIR" || exit 1

    # 执行编译
    local build_cmd="bash build.sh -n $OP_NAME -c $DEVICE_TYPE -p $CANN_ENV --ops-compile-options \"$SANITIZER_OPTS\""
    log_info "执行编译: $build_cmd"
    echo "outdir: $OUTPUT_DIR/build/build.log"

    if eval "$build_cmd" > "$OUTPUT_DIR/build/build.log" 2>&1; then
        log_success "编译成功"
    else
        log_error "编译失败"
        # 提取错误信息
        tail -50 "$OUTPUT_DIR/build/build.log" | tee "$OUTPUT_DIR/build/build_errors.log"
        exit 1
    fi

    if [ "$VERBOSE" = true ]; then
        log_verbose "编译日志已保存到: $OUTPUT_DIR/build/build.log"
    fi

    # 检查编译产物
    local package_dir="$CODE_BASE_DIR/output"
    local install_package=$(find "$package_dir" -name "*.run" | head -1)

    if [ -z "$install_package" ]; then
        log_error "未找到编译产物（.run 文件）: $package_dir"
        exit 1
    fi

    log_success "编译产物: $install_package"

    # 保存编译产物路径
    echo "$install_package" > "$OUTPUT_DIR/build/package_path.txt"
}

# ============================================================================
# 第 2 步：安装算子
# ============================================================================

install_operator() {
    log_info "===== 第 2 步：安装算子 ====="

    # 查找编译产物
    local package_dir="$CODE_BASE_DIR/output"
    local install_package=$(find "$package_dir" -name "*.run" | head -1)

    if [ -z "$install_package" ]; then
        log_error "未找到编译产物（.run 文件）: $package_dir"
        log_error "请先执行编译步骤"
        exit 1
    fi

    log_info "安装包: $install_package"

    # 创建安装目录
    if [ ! -d "$INSTALL_PATH" ]; then
        log_info "创建安装目录: $INSTALL_PATH"
        mkdir -p "$INSTALL_PATH"
    else
        log_info "清理旧安装: $INSTALL_PATH"
        rm -rf "$INSTALL_PATH"
        log_info "重新创建安装目录: $INSTALL_PATH"
        mkdir -p "$INSTALL_PATH"
    fi

    # 执行安装
    log_info "安装到: $INSTALL_PATH"
    if "$install_package" --install-path="$INSTALL_PATH" > "$OUTPUT_DIR/install/install.log" 2>&1; then
        log_success "安装成功"
    else
        log_error "安装失败"
        tail -50 "$OUTPUT_DIR/install/install.log" | tee "$OUTPUT_DIR/install/install_errors.log"
        exit 1
    fi

    if [ "$VERBOSE" = true ]; then
        log_verbose "安装日志已保存到: $OUTPUT_DIR/install/install.log"
    fi

    # 加载算子环境
    if [ "$LOAD_ENV" = true ]; then
        local env_script=$(find "$INSTALL_PATH/vendors" -path "*/bin/set_env.bash" -print -quit 2>/dev/null)
        if [ -f "$env_script" ]; then
            log_info "加载算子环境: $env_script"
            source "$env_script"
            log_success "算子环境已加载 source $env_script"
        else
            log_warning "算子环境脚本未找到: $env_script"
            log_warning "可能需要手动加载环境"
        fi
    fi
}

# ============================================================================
# 第 3 步：运行 MemCheck
# ============================================================================

run_memcheck() {
    log_info "===== 第 3 步：运行 MemCheck ====="

    # 使用绝对路径获取测试脚本目录和名称
    local test_script_dir="$TEST_SCRIPT_DIR"

    # 进入测试目录
    if [ ! -d "$test_script_dir" ]; then
        log_error "测试目录不存在: $test_script_dir"
        exit 1
    fi

    log_info "进入测试目录: $test_script_dir"
    cd "$test_script_dir" || exit 1

    # 设置环境变量
    log_info "设置环境变量:"
    log_info "  ASCEND_GLOBAL_LOG_LEVEL=$LOG_LEVEL"
    log_info "  ASCEND_SLOG_PRINT_TO_STDOUT=$SLOG_PRINT"

    export ASCEND_GLOBAL_LOG_LEVEL="$LOG_LEVEL"
    export ASCEND_SLOG_PRINT_TO_STDOUT="$SLOG_PRINT"

    # 运行 memcheck
    local memcheck_cmd="mssanitizer --tool=memcheck $TEST_SCRIPT_EXE"
    log_info "执行 Memcheck: $memcheck_cmd"

    # 保存原始报告
    local raw_report="$OUTPUT_DIR/memcheck/ascendc_memcheck_report_raw.txt"

    if eval "$memcheck_cmd" > "$raw_report" 2>&1; then
        log_success "Memcheck 执行完成"
    else
        log_warning "Memcheck 执行完成（可能检测到错误）"
    fi

    # 保存完整日志
    cp "$raw_report" "$OUTPUT_DIR/memcheck/memcheck.log"

    # 查找并复制 mindstudio_sanitizer_log 目录
    if [ -d "mindstudio_sanitizer_log" ]; then
        cp -r mindstudio_sanitizer_log "$OUTPUT_DIR/memcheck/"
        log_success "Sanitizer 日志已复制到: $OUTPUT_DIR/memcheck/mindstudio_sanitizer_log"
    fi

    # 提取报告摘要
    log_info "===== Memcheck 结果摘要 ====="

    local error_count=$(grep -c "====== ERROR:" "$raw_report" 2>/dev/null || echo "0")
    local warning_count=$(grep -c "====== WARNING:" "$raw_report" 2>/dev/null || echo "0")

    echo "ERROR 数量: $error_count"
    echo "WARNING 数量: $warning_count"

    if [ "$error_count" -gt 0 ]; then
        grep "====== ERROR:" "$raw_report" | head -5
    fi

    if [ "$warning_count" -gt 0 ]; then
        grep "====== WARNING:" "$raw_report" | head -5
    fi

    # 检查测试结果
    grep -E "collected|passed|FAILED" "$raw_report" | tail -10
}

# ============================================================================
# 生成状态报告
# ============================================================================

generate_status_report() {
    log_info "===== 生成状态报告 ====="

    local status_file="$OUTPUT_DIR/status.txt"

    cat > "$status_file" << EOF
AscendC Memcheck 执行状态报告
================================

执行时间: $(date)
配置文件: $CONFIG_FILE

执行步骤:
EOF

    # 编译状态
    if [ "$SKIP_BUILD" = true ] || [ "$REBUILD" = false ]; then
        echo "  第 1 步（编译）: 跳过" >> "$status_file"
    else
        if [ -f "$OUTPUT_DIR/build/build.log" ]; then
            echo "  第 1 步（编译）: 成功" >> "$status_file"
        else
            echo "  第 1 步（编译）: 失败" >> "$status_file"
        fi
    fi

    # 安装状态
    if [ -f "$OUTPUT_DIR/install/install.log" ]; then
        echo "  第 2 步（安装）: 成功" >> "$status_file"
    else
        echo "  第 2 步（安装）: 失败" >> "$status_file"
    fi

    # Memcheck 状态
    if [ -f "$OUTPUT_DIR/memcheck/memcheck.log" ]; then
        echo "  第 3 步（Memcheck）: 成功" >> "$status_file"
    else
        echo "  第 3 步（Memcheck）: 失败" >> "$status_file"
    fi

    echo "" >> "$status_file"
    echo "输出文件:" >> "$status_file"
    echo "  构建日志: $OUTPUT_DIR/build/build.log" >> "$status_file"
    echo "  安装日志: $OUTPUT_DIR/install/install.log" >> "$status_file"
    echo "  Memcheck 日志: $OUTPUT_DIR/memcheck/memcheck.log" >> "$status_file"
    echo "  原始报告: $OUTPUT_DIR/memcheck/ascendc_memcheck_report_raw.txt"
    echo "" >> "$status_file"

    # 保存时间戳
    date > "$OUTPUT_DIR/timestamp.txt"

    log_success "状态报告已生成: $status_file"
    cat "$status_file"
}

# ============================================================================
# 清理构建目录
# ============================================================================

cleanup_build() {
    if [ "$KEEP_BUILD" = false ]; then
        log_info "清理构建目录（可选）"
        # 这里可以添加清理逻辑
        # 例如：cd "$CODE_BASE_DIR" && rm -rf build/
    fi
}

# ============================================================================
# 主函数
# ============================================================================

main() {
    log_info "===== AscendC Memcheck 自动化脚本 ====="
    log_info "开始时间: $(date)"
    log_info ""

    # 解析参数
    parse_arguments "$@"

    # 解析配置
    parse_config "$CONFIG_FILE"

    # 验证环境
    check_environment

    # 执行各步骤
    build_operator
    install_operator
    run_memcheck

    # 生成报告
    generate_status_report

    # 清理
    cleanup_build

    log_info ""
    log_info "===== 执行完成 ====="
    log_info "结束时间: $(date)"
    log_info "输出目录: $OUTPUT_DIR"
    log_success "所有步骤执行完成"
}

# ============================================================================
# 脚本入口
# ============================================================================

main "$@"
