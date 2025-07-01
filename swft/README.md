# SWFT

## SWFT 简介
SWFT是一款Ascend算子编译器，有着极简编写、灵活切分等特征, 旨在帮助开发者快速实现各类计算图算子，降低算子开发门槛，提升开发效率。

## 功能特点
Ascend亲和：昇腾指令集亲和的接口设计
大范围融合支持：支持大范围自定义融合算子生成
灵活切分设计：支持前端便捷表达切分方式
Python原生：支持Python侧完成开发算子、验证精度、测试性能等流程

## 目录结构
```shell
swft/
├── python/                     # 核心生成引擎
├────swft/
│      ├── utils/               # 合法性检查接口
│      ├── intrinsic/           # 具体指令类实现
│      ├── core/                # 核心编译模块
│      └── api/                 # dsl接口
├── lib/                        # 后端codegen
├── docs/                       # 文档目录
│   ├── slicedata.md            # 数据切分api文档
│   ├── move.md                 # 数据搬运api文档
│   ├── compute.md              # 计算api文档
│   └── composite.md            # 复合算子api文档
├── pybind/                     # Pybind相关
├── tests/                      # 测试用例
├── pyproject.toml              # 打包相关配置
├── setup.cfg                   # 打包相关配置
├── build.sh                    # 构建脚本
└── README.md                   # 项目说明文档

```
相关文档链接：
[slicedata.md](docs/slicedata.md)
[move.md](docs/move.md)
[compute.md](docs/compute.md)
[composite.md](docs/composite.md)

## 快速开始
### 环境要求
1. gcc>=9.4
2. python>=3.7
3. pybind11
4. setuptools>=42
5. build>=1.2.2.post1
6. wheel>=0.45.1

### 安装方法
自动打包和安装swft软件包：
```shell
bash build.sh
```
### 使用示例
```shell
python op_tests/fusion/adam.py
```
## 致谢
感谢以下项目对本工具的启发和支持：
    MindStudio Kernel Performance Prediction (MsKPP)
    https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/devaids/optool/atlasopdev_16_0006.html
