#!/bin/bash

# 设置错误时退出
set -e

# 解析命令行参数
TRITON_ASCEND=false
BUILD_WHEEL=false

show_help() {
    echo "用法: ./build.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --triton_ascend    安装 triton-ascend 相关依赖并以开发模式安装"
    echo "  --wheel            构建 wheel 包（默认行为，保持向后兼容）"
    echo "  -h, --help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./build.sh                    # 构建 wheel 包"
    echo "  ./build.sh --wheel            # 构建 wheel 包"
    echo "  ./build.sh --triton_ascend    # 安装 triton-ascend 依赖并开发模式安装"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --triton_ascend)
            TRITON_ASCEND=true
            shift
            ;;
        --wheel)
            BUILD_WHEEL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果没有指定任何选项，默认构建 wheel
if [ "$TRITON_ASCEND" = false ] && [ "$BUILD_WHEEL" = false ]; then
    BUILD_WHEEL=true
fi

# Triton Ascend 开发模式安装
if [ "$TRITON_ASCEND" = true ]; then
    echo "================================================"
    echo "开始 Triton Ascend 开发环境安装..."
    echo "================================================"
    
    # 安装 triton-ascend 相关依赖
    echo "安装 requirements_triton_ascend.txt 依赖..."
    pip install -r requirements_triton_ascend.txt
    
    # 开发模式安装
    echo "以开发模式安装 ai_kernel_generator..."
    pip install -e ./
    
    echo ""
    echo "================================================"
    echo "Triton Ascend 开发环境安装完成！"
    echo "================================================"
    echo ""
    echo "提示: 使用 pip install -e ./ 后，不再需要 source env.sh"
    echo "现在可以直接使用 ai_kernel_generator 了。"
fi

# 构建 wheel 包
if [ "$BUILD_WHEEL" = true ]; then
    echo "================================================"
    echo "开始构建 wheel 包..."
    echo "================================================"
    
    # 清理旧的构建文件
    echo "清理旧的构建文件..."
    rm -rf build/ dist/ *.egg-info/

    # 创建输出目录
    echo "创建输出目录..."
    mkdir -p output

    # 构建 wheel 包
    echo "构建 wheel 包..."
    python setup.py bdist_wheel --dist-dir output

    # 显示构建结果
    echo ""
    echo "================================================"
    echo "构建完成！"
    echo "================================================"
    echo "wheel 包位置:"
    ls -l output/

    # 可选：显示安装命令提示
    echo -e "\n您可以使用以下命令安装包："
    echo "pip install output/ai_kernel_generator-*-py3-none-any.whl"
fi
