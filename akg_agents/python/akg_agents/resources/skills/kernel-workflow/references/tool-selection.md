# 子Agent 选择指南

## 选择逻辑

1. **用户明确指定** → 根据关键词识别
2. **用户未指定** → 使用 `ask_user` 询问
3. **调用前二次确认** → **必须**再次 `ask_user` 确认用户选择

## 关键词识别

1. "用 evolve"、"进化搜索"、"高性能" → `call_evolve`
2. "用 coder_only"、"快速生成" → `call_coder_only`
3. "用 adaptive_search"、"树搜索" → `call_adaptive_search`
4. "测试性能"、"profile" → `call_coder_only(task_type="profile")`

## 询问用户

用户未指定时：

```python
ask_user(message="请选择生成方式：\n1. coder_only - 快速生成\n2. evolve - 进化搜索（高性能）\n3. adaptive_search - 树搜索")
```

## 二次确认（必须）

**在调用子Agent 前，必须再次询问用户确认：**

```python
ask_user(message="即将使用 [选择的方式] 生成 kernel，确认执行吗？(y/n)")
```

- 用户回复 "y"/"yes"/"确认" → 执行调用
- 用户回复 "n"/"no"/"取消" 或其他方式 → 重新询问选择

## 子Agent 参数

1. `call_coder_only(task_code="...", op_name="relu", user_requirements="...")`
   - task_type: "precision_only"（默认）或 "profile"
   - user_requirements: kernel 优化需求（可选，见下方说明）

2. `call_evolve(task_code="...", op_name="relu", user_requirements="...")`

3. `call_adaptive_search(task_code="...", op_name="relu", user_requirements="...")`

4. `call_op_task_builder(user_request="...")`

5. `call_kernel_verifier(kernel_code="...", task_code="...", op_name="relu")`

---

## user_requirements 参数说明（重要！）

### 需要传递 user_requirements 的情况

**只有"对 kernel 实现的要求"才需要传递 `user_requirements`**，这些需求会传递给子Agent，影响最终生成的 kernel 代码：

1. 切分策略: "核内进行二次切分"、"使用分块策略" 
2. 硬件特性: "使用 tensor core"、"利用 shared memory"


**示例：**
```
用户: "生成 ReLU 算子，核内进行二次切分"
   → call_coder_only(task_code="...", op_name="relu", user_requirements="核内进行二次切分")
```

### 不需要传递 user_requirements 的情况

**用户没有传递对kernel生成的要求**
**"对 task 代码的要求"不传递 `user_requirements`**，而是前面已经通过 `call_op_task_builder` 处理完成了。

## 禁止行为

1. 用户未指定时自动选择
2. 代码不完整直接调用子Agent
3. 不进行二次确认直接调用子Agent
4. **把 task 相关需求（数据类型、shape）放入 user_requirements**
