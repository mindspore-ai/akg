#!/bin/bash

# AIKG测试运行脚本 - 用于CI环境
# 用法: ./run_test.sh -t op-ut
#       ./run_test.sh -t op-st -m "torch and triton and cuda and a100"

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << 'EOF'
AIKG测试运行脚本

用法:
    ./run_test.sh [-t test_type] [-m "marker_expression"] [其他pytest参数]

测试类型 (-t):
    ut          核心基础设施单元测试  (tests/ut/)     - 不需要LLM/GPU
    st          核心基础设施系统测试  (tests/st/)     - 需要LLM
    op-ut       算子相关单元测试      (tests/op/ut/)  - 不需要LLM/GPU
    op-st       算子相关系统测试      (tests/op/st/)  - 需要LLM或设备
    op-bench    算子Benchmark测试     (tests/op/bench/) - 需要LLM和设备

可选参数:
    -m "marker_expression"     pytest标记表达式
    -t test_type              测试类型（默认: op-ut）
    -h, --help                显示此帮助信息
    -v, --verbose             详细输出
    -k "test_name"            按测试名称过滤
    -x                        遇到第一个失败就停止
    --tb=style                回溯格式 (auto/long/short/line/no)
    --maxfail=num             最大失败数量

示例:
    # 运行核心基础设施UT（不需要任何外部依赖）
    ./run_test.sh -t ut

    # 运行算子UT（不需要LLM/GPU）
    ./run_test.sh -t op-ut
    ./run_test.sh -t op-ut -m "level0"

    # 运行CI测试（无LLM/向量库依赖）
    ./run_test.sh -t op-ut -m "level0 and not use_model and not use_vector_store"

    # 运行算子ST（需要LLM和/或设备）
    ./run_test.sh -t op-st -m "torch and triton and cuda and a100"
    ./run_test.sh -t op-st -m "torch and triton and ascend and ascend910b4"

    # 运行算子Benchmark
    ./run_test.sh -t op-bench -m "torch and triton and cuda and a100"
    ./run_test.sh -t op-bench -m "torch and triton and ascend and ascend910b4"

    # 运行核心ST（需要LLM）
    ./run_test.sh -t st

注意事项:
    1. op-st 和 op-bench 需要包含完整的硬件配置标记（framework、dsl、backend、arch）
    2. ut 和 op-ut 不需要硬件标记
    3. 可以使用 and、or、not 等逻辑操作符组合标记

EOF
}

# 验证标记表达式是否包含必要的配置信息
validate_markers() {
    local markers="$1"
    local test_type="$2"

    # 定义硬件配置标记列表
    local framework_markers=("torch" "mindspore" "numpy")
    local dsl_markers=("triton" "ascendc" "cpp" "cuda_c" "tilelang" "tilelang_npuir")
    local backend_markers=("cuda" "ascend" "cpu")
    local arch_markers=("a100" "v100" "h20" "l20" "rtx3090" "ascend910b4" "ascend910b2" "ascend910_9362" "ascend910_9372" "ascend910_9381" "ascend910_9382" "ascend910_9391" "ascend910_9392" "ascend950dt_95a" "ascend950pr_950z" "ascend950pr_9572" "ascend950pr_9574" "ascend950pr_9575" "ascend950pr_9576" "ascend950pr_9577" "ascend950pr_9578" "ascend950pr_9579" "ascend950pr_957b" "ascend950pr_957d" "ascend950pr_9581" "ascend950pr_9582" "ascend950pr_9584" "ascend950pr_9587" "ascend950pr_9588" "ascend950pr_9589" "ascend950pr_958a" "ascend950pr_958b" "ascend950pr_9591" "ascend950pr_9592" "ascend950pr_9599" "x86_64")

    # ut 和 op-ut 不需要硬件标记验证
    case "$test_type" in
        ut|op-ut)
            if [[ -n "$markers" ]]; then
                print_info "使用标记表达式: '$markers'"
            else
                print_info "$test_type 测试无标记，将运行全量测试"
            fi
            return 0
            ;;
        st)
            # 核心st也不强制硬件标记
            if [[ -n "$markers" ]]; then
                print_info "使用标记表达式: '$markers'"
            fi
            return 0
            ;;
    esac

    # op-st 和 op-bench 需要硬件标记
    if [[ -z "$markers" ]]; then
        print_error "$test_type 测试必须指定标记表达式！"
        print_error "例如: -m 'torch and triton and cuda and a100'"
        exit 1
    fi

    local has_framework=false has_dsl=false has_backend=false has_arch=false

    for marker in "${framework_markers[@]}"; do
        [[ "$markers" == *"$marker"* ]] && has_framework=true && break
    done
    for marker in "${dsl_markers[@]}"; do
        [[ "$markers" == *"$marker"* ]] && has_dsl=true && break
    done
    for marker in "${backend_markers[@]}"; do
        [[ "$markers" == *"$marker"* ]] && has_backend=true && break
    done
    for marker in "${arch_markers[@]}"; do
        [[ "$markers" == *"$marker"* ]] && has_arch=true && break
    done

    local marker_count=0
    [[ "$has_framework" == true ]] && marker_count=$((marker_count + 1))
    [[ "$has_dsl" == true ]] && marker_count=$((marker_count + 1))
    [[ "$has_backend" == true ]] && marker_count=$((marker_count + 1))
    [[ "$has_arch" == true ]] && marker_count=$((marker_count + 1))

    if [[ $marker_count -lt 4 ]]; then
        print_error "标记表达式 '$markers' 配置信息不足！"
        print_error "必须包含所有4个标记: framework、dsl、backend、arch"
        print_error "  framework: ${framework_markers[*]}"
        print_error "  dsl:       ${dsl_markers[*]}"
        print_error "  backend:   ${backend_markers[*]}"
        print_error "  arch:      ${arch_markers[*]}"
        print_error "例如: -m 'torch and triton and cuda and a100'"
        exit 1
    fi

    print_info "标记验证通过 (framework=$has_framework, dsl=$has_dsl, backend=$has_backend, arch=$has_arch)"
}

# 获取测试路径
get_test_path() {
    local test_type="$1"

    case "$test_type" in
        ut)       echo "tests/ut/" ;;
        st)       echo "tests/st/" ;;
        op-ut)    echo "tests/op/ut/" ;;
        op-st)    echo "tests/op/st/" ;;
        op-bench) echo "tests/op/bench/" ;;
        *)
            print_error "未知的测试类型: $test_type"
            print_error "支持的类型: ut, st, op-ut, op-st, op-bench"
            exit 1
            ;;
    esac
}

# 检查pytest是否可用
check_pytest() {
    if ! command -v pytest &> /dev/null; then
        print_error "pytest 未安装或不在PATH中！"
        print_error "请安装pytest: pip install pytest"
        exit 1
    fi
    print_info "pytest版本: $(pytest --version 2>&1 | head -1)"
}

# 检查是否在正确的目录
check_directory() {
    if [[ ! -d "tests" ]]; then
        print_error "当前目录下没有找到 tests 目录！"
        print_error "请确保在 akg_agents 目录下运行此脚本"
        print_error "当前目录: $(pwd)"
        exit 1
    fi
}

# 主函数
main() {
    local markers=""
    local test_type="op-ut"  # 默认运行op-ut测试
    local pytest_args=()

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m)
                markers="$2"
                shift 2
                ;;
            -t)
                test_type="$2"
                shift 2
                ;;
            *)
                pytest_args+=("$1")
                shift
                ;;
        esac
    done

    # 显示脚本信息
    echo "=========================================="
    echo "    AIKG测试运行脚本"
    echo "=========================================="
    echo

    # 验证参数
    validate_markers "$markers" "$test_type"

    # 检查环境
    check_pytest
    check_directory

    # 获取测试路径
    local test_path
    test_path=$(get_test_path "$test_type")

    # 构建pytest命令
    local cmd=(pytest -sv --disable-warnings "$test_path")

    if [[ -n "$markers" ]]; then
        cmd+=(-m "$markers")
    fi
    if [[ ${#pytest_args[@]} -gt 0 ]]; then
        cmd+=("${pytest_args[@]}")
    fi

    # 打印执行信息
    print_info "测试类型: $test_type"
    print_info "测试路径: $test_path"
    if [[ -n "$markers" ]]; then
        print_info "标记表达式: $markers"
    fi
    local pretty_cmd=""
    printf -v pretty_cmd "%q " "${cmd[@]}"
    print_info "执行命令: ${pretty_cmd}"
    echo

    # 运行测试
    echo "=========================================="
    if "${cmd[@]}"; then
        echo "=========================================="
        print_success "所有测试通过！"
        exit 0
    else
        echo "=========================================="
        print_error "测试失败！"
        exit 1
    fi
}

# 如果脚本被直接执行，则运行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
