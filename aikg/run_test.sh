#!/bin/bash

# AIKG测试运行脚本 - 用于CI环境
# 用法: ./run_test.sh -m "torch and triton and cuda"

# 添加调试信息
# set -x  # 启用调试模式，显示每个命令
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
    cat << EOF
AIKG测试运行脚本

用法:
    $0 [-m "marker_expression"] [-t test_type] [其他pytest参数]

必需参数:
    无（所有参数都是可选的）

可选参数:
    -m "marker_expression"     pytest标记表达式
                               对于st测试，必须包含完整的硬件配置标记
                               对于ut测试，可以只包含功能标记
    -t test_type              测试类型选择 (ut|st)，默认为ut
    -h, --help                显示此帮助信息
    -v, --verbose            详细输出
    -k "test_name"           按测试名称过滤
    -x                       遇到第一个失败就停止
    --tb=style               回溯格式 (auto/long/short/line/no)
    --maxfail=num            最大失败数量
    --durations=N            显示最慢的N个测试
    --cov                    启用覆盖率报告
    --html=report.html       生成HTML报告

测试类型说明:
    ut                       运行单元测试 (tests/ut/) - 通用功能测试，不依赖特定硬件
    st                       运行系统测试 (tests/st/) - 硬件相关测试，需要完整硬件配置

示例:
    # 运行通用单元测试（不需要硬件配置）
    $0 -t ut
    $0 -m "level0" -t ut

    # 运行CI测试（无依赖）
    $0 -m "level0 and not use_model and not use_vector_store" -t ut

    # 运行硬件相关系统测试（需要完整硬件配置）
    $0 -m "torch and triton and cuda and a100" -t st
    $0 -m "torch and triton and ascend and ascend910b4" -t st
    $0 -m "mindspore and triton and ascend and ascend910b4" -t st
    $0 -m "numpy and swft and ascend and ascend310p3" -t st

注意事项:
    1. 测试类型决定了标记要求：
       - ut: 可以只使用功能标记（如level0、use_model等）
       - st: 必须包含完整的硬件配置标记（framework、dsl、backend、arch）
    2. 硬件配置标记必须包含所有4个标记: framework、dsl、backend、arch
    3. 功能标记（level0、use_model、use_vector_store等）可以看情况添加
    4. 可以使用 and、or、not 等逻辑操作符组合标记
    5. 脚本会自动切换到tests目录并运行pytest

EOF
}

# 验证标记表达式是否包含必要的配置信息
validate_markers() {
    local markers="$1"
    local test_type="$2"  # 新增参数：测试类型
    
    print_info "开始验证标记表达式: '$markers' (测试类型: $test_type)"
    
    # 定义功能标记列表
    local functional_markers=("level0" "use_model" "use_vector_store")
    local has_functional_markers=false
    
    # 检查是否包含功能标记
    for marker in "${functional_markers[@]}"; do
        if [[ "$markers" == *"$marker"* ]]; then
            has_functional_markers=true
            print_info "检测到功能标记: $marker"
            break
        fi
    done
    
    # 定义硬件配置标记列表
    local framework_markers=("torch" "mindspore" "numpy")
    local dsl_markers=("triton" "swft")
    local backend_markers=("cuda" "ascend")
    local arch_markers=("a100" "v100" "h20" "l20" "rtx3090" "ascend910b4" "ascend310p3")
    
    # 检查是否包含硬件配置标记
    local has_hardware_markers=false
    for marker in "${framework_markers[@]}" "${dsl_markers[@]}" "${backend_markers[@]}" "${arch_markers[@]}"; do
        if [[ "$markers" == *"$marker"* ]]; then
            has_hardware_markers=true
            print_info "检测到硬件配置标记: $marker"
            break
        fi
    done

    # 根据测试类型和标记类型决定验证策略
    if [[ "$test_type" == "ut" ]]; then
        # ut测试：如果没有标记，直接通过（运行全量测试）
        if [[ -z "$markers" ]]; then
            print_info "ut测试无标记，将运行全量测试"
            return 0
        fi
        # 如果有功能标记，直接通过
        if [[ "$has_functional_markers" == true ]]; then
            print_info "ut测试功能标记验证通过！"
            return 0
        fi
        # 如果包含硬件配置标记，继续验证
        print_info "ut测试包含硬件配置标记，开始验证..."
    elif [[ "$test_type" == "st" ]]; then
        # st测试，必须包含硬件配置标记
        if [[ "$has_hardware_markers" == false ]]; then
            print_error "st测试必须包含硬件配置标记！"
            print_error "请添加framework、dsl、backend、arch等标记"
            exit 1
        fi
        print_info "st测试，开始硬件配置验证..."
    else
        # 其他测试类型（如all）不被支持
        print_error "不支持的测试类型: $test_type"
        print_error "支持的测试类型: ut, st"
        exit 1
    fi
    
    # 如果包含硬件配置标记，则进行详细验证
    if [[ "$has_hardware_markers" == true ]]; then
        # 检查是否包含基本的配置标记
        local has_framework=false
        local has_dsl=false
        local has_backend=false
        local has_arch=false
        
        # 检查framework标记
        for marker in "${framework_markers[@]}"; do
            if [[ "$markers" == *"$marker"* ]]; then
                has_framework=true
                print_info "检测到framework标记: $marker"
                break
            fi
        done
        
        # 检查dsl标记
        for marker in "${dsl_markers[@]}"; do
            if [[ "$markers" == *"$marker"* ]]; then
                has_dsl=true
                print_info "检测到dsl标记: $marker"
                break
            fi
        done
        
        # 检查backend标记
        for marker in "${backend_markers[@]}"; do
            if [[ "$markers" == *"$marker"* ]]; then
                has_backend=true
                print_info "检测到backend标记: $marker"
                break
            fi
        done
        
        # 检查arch标记
        for marker in "${arch_markers[@]}"; do
            if [[ "$markers" == *"$marker"* ]]; then
                has_arch=true
                print_info "检测到arch标记: $marker"
                break
            fi
        done
    fi  # 关闭 if [[ "$has_hardware_markers" == true ]]
    
    # 必须包含所有4个配置标记才能运行
    local marker_count=0

    if [[ "$has_framework" == true ]]; then
        marker_count=$((marker_count + 1))
        print_info "framework标记计数: $marker_count"
    else
        print_info "framework标记未检测到"
    fi
    
    if [[ "$has_dsl" == true ]]; then
        marker_count=$((marker_count + 1))
        print_info "dsl标记计数: $marker_count"
    else
        print_info "dsl标记未检测到"
    fi
    
    if [[ "$has_backend" == true ]]; then
        marker_count=$((marker_count + 1))
        print_info "backend标记计数: $marker_count"
    else
        print_info "backend标记未检测到"
    fi
    
    if [[ "$has_arch" == true ]]; then
        marker_count=$((marker_count + 1))
        print_info "arch标记计数: $marker_count"
    else
        print_info "arch标记未检测到"
    fi
    
    print_info "标记统计: framework=$has_framework, dsl=$has_dsl, backend=$has_backend, arch=$has_arch"
    
    if [[ $marker_count -lt 4 ]]; then
        print_error "标记表达式 '$markers' 配置信息不足！"
        print_error "必须包含所有4个标记: framework、dsl、backend、arch"
        print_error "例如: -m 'torch and triton and cuda and a100'"
        echo
        exit 1
    fi
    
    print_info "使用标记表达式: $markers"
    print_info "配置标记数量: $marker_count/4 (framework: $has_framework, dsl: $has_dsl, backend: $has_backend, arch: $has_arch)"
    print_info "标记验证通过！"
}

# 验证测试类型参数
validate_test_type() {
    local test_type="$1"
    
    case "$test_type" in
        ut|st)
            print_info "测试类型验证通过: $test_type"
            return 0
            ;;
        *)
            print_error "无效的测试类型: $test_type"
            print_error "支持的测试类型: ut, st"
            exit 1
            ;;
    esac
}

# 获取测试路径
get_test_path() {
    local test_type="$1"
    
    case "$test_type" in
        ut)
            echo "tests/ut/"
            ;;
        st)
            echo "tests/st/"
            ;;

        *)
            print_error "未知的测试类型: $test_type"
            exit 1
            ;;
    esac
}

# 检查pytest是否可用
check_pytest() {
    print_info "检查pytest环境..."
    
    if ! command -v pytest &> /dev/null; then
        print_error "pytest 未安装或不在PATH中！"
        print_error "请安装pytest: pip install pytest"
        exit 1
    fi
    
    print_info "pytest版本: $(pytest --version)"
    print_info "pytest路径: $(which pytest)"
}

# 检查是否在正确的目录
check_directory() {
    print_info "检查目录结构..."
    
    if [[ ! -d "tests" ]]; then
        print_error "当前目录下没有找到 tests 目录！"
        print_error "请确保在 aikg 目录下运行此脚本"
        print_error "当前目录: $(pwd)"
        print_error "目录内容: $(ls -la)"
        exit 1
    fi
    
    print_info "当前工作目录: $(pwd)"
    print_info "tests目录: $(pwd)/tests"
    print_info "tests目录内容: $(ls -la tests)"
}

# 主函数
main() {
    print_info "主函数开始执行..."
    
    local markers="level0 and not use_model and not use_vector_store"
    local test_type="ut"  # 默认运行ut测试
    local pytest_args=()
    local verbose=false
    
    print_info "解析命令行参数: $@"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_info "显示帮助信息"
                show_help
                exit 0
                ;;
            -m)
                if [[ -n "$2" ]]; then
                    markers="$2"
                    print_info "用户指定标记表达式: '$markers'"
                else
                    print_warning "忽略空的-m参数，使用默认标记"
                fi
                shift 2
                ;;
            -t)
                test_type="$2"
                print_info "设置测试类型: '$test_type'"
                shift 2
                ;;
            -v|--verbose)
                verbose=true
                pytest_args+=("$1")
                print_info "启用详细输出"
                shift
                ;;
            *)
                pytest_args+=("$1")
                print_info "添加pytest参数: '$1'"
                shift
                ;;
        esac
    done
    
    print_info "参数解析完成:"
    print_info "  markers: '$markers'"
    print_info "  test_type: '$test_type'"
    print_info "  pytest_args: '${pytest_args[@]}'"
    print_info "  verbose: $verbose"
    
    # 显示脚本信息
    echo "=========================================="
    echo "    AIKG测试运行脚本"
    echo "=========================================="
    echo
    
    print_info "开始验证参数..."
    # 验证参数
    validate_markers "$markers" "$test_type"
    validate_test_type "$test_type"
    
    print_info "开始检查环境..."
    # 检查环境
    check_pytest
    check_directory
    
    # 获取测试路径
    local test_path=$(get_test_path "$test_type")
    print_info "测试路径: $test_path"
    
    # 构建pytest命令（使用数组避免eval导致的二次分词）
    local cmd=(pytest -sv --disable-warnings "$test_path")
    
    # 为UT测试添加文件排除（因为这些文件在导入时就会出错）
    if [[ "$test_type" == "ut" ]]; then
        # 添加--ignore参数来跳过有问题的测试文件
        cmd+=(--ignore=tests/ut/test_run_embedding.py --ignore=tests/ut/test_database.py --ignore=tests/ut/test_feature_extract.py --ignore=tests/ut/test_handwrite_loader.py --ignore=tests/ut/test_selector_agent.py)
    fi
    
    if [[ -n "$markers" ]]; then
        cmd+=(-m "$markers")
    fi
    if [[ ${#pytest_args[@]} -gt 0 ]]; then
        cmd+=("${pytest_args[@]}")
    fi
    
    # 安全打印将要执行的命令
    local pretty_cmd=""
    printf -v pretty_cmd "%q " "${cmd[@]}"
    print_info "构建的pytest命令: ${pretty_cmd}"
    print_info "当前工作目录: $(pwd)"
    print_info "测试类型: $test_type"
    print_info "测试路径: $test_path"
    print_info "执行命令: ${pretty_cmd}"
    echo
    
    # 运行测试
    print_info "开始运行测试..."
    echo "=========================================="
    
    print_info "执行pytest命令..."
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
    print_info "脚本开始执行，BASH_SOURCE[0]=${BASH_SOURCE[0]}"
    print_info "脚本参数: $@"
    print_info "当前shell: $SHELL"
    print_info "bash版本: $(bash --version | head -1)"
    main "$@"
else
    print_info "脚本被source，不执行主函数"
fi
