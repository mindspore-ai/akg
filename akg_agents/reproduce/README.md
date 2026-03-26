# reproduce — 复现脚本目录

本目录提供让外部用户复现 AKG Agents 流程效果的独立脚本。

## 目录结构

```
reproduce/
├── SPEC.md              # 本目录开发规范
├── README.md            # 本文件
├── wip/                 # 临时建设中的脚本（不保证完全验证）
└── (复现脚本)           # 已验证的复现脚本将放在根目录
```

## 使用说明

### 运行前准备

1. 确保已安装 akg_agents：
   ```bash
   pip install -e ./ --no-build-isolation
   # 或
   source env.sh
   ```

2. 配置必要的环境变量（如 API Key）：
   ```bash
   export OPENAI_API_KEY=your_key_here
   # 或在 ~/.akg_agents/settings.json 中配置
   ```

### 运行脚本

每个脚本都支持 `--help` 查看详细参数：

```bash
python reproduce/<script_name>.py --help
```

## wip/ 子目录

`wip/` 目录存放**临时建设中的脚本**，这些脚本：
- 可能尚未完全验证
- 可能包含实验性功能
- 文档可能不完整

验证通过后会移至 `reproduce/` 根目录。

## 贡献指南

添加新的复现脚本时，请确保：
1. 脚本头部或配套 README 说明复现目标、前置条件、运行方式
2. 支持 `--help` 参数
3. 输出清晰，关键指标可视化
4. 先放在 `wip/` 验证，通过后移至根目录
