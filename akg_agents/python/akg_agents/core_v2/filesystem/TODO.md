# FileSystem (core_v2) 改进 TODO

> 创建日期: 2026-02-10
> 最后更新: 2026-02-10
> 状态: 进行中

---

## 🔴 P0 - 高优先级

### 1. ✅ 跨节点代码演化追踪 (blame) — 已完成 (2026-02-10)
- **实现**: `TraceSystem.blame_file()` + `TraceSystem.blame_all_files()`
- **位置**: `trace_system.py` (末尾新增 ~100 行)
- **测试**: `test_blame.py` — 10 个用例全部通过
  - 线性路径创建/修改、root 节点文件、文件不存在
  - 分叉独立演化、文件删除检测、多次修改
  - blame_all_files 多文件追踪、空路径、同内容覆写去重
- [x] 实现 blame_file 方法
- [x] 实现 blame_all_files 方法
- [x] 添加单元测试 (10/10 通过)
- [ ] (可选) 集成到 CLI `/trace blame <file>` 命令

### 2. ✅ 三路合并算法完善 — 已完成 (2026-02-10)
- **实现**: 手写基于 `difflib` 的三路合并算法 (~80 行核心代码)
- **特点**: **零外部依赖** (移除了 `merge3` 库)，完全利用标准库实现
- **能力**:
  - 自动合并非重叠区域的修改
  - 仅在真正重叠修改时标记冲突
  - 兼容 git 风格的冲突标记 (`<<<<<<< YOURS`)
- **测试**: `test_filesystem_diff_merge.py` — 8 个用例全部通过
  - 新增: 自动合并不同区域、局部冲突、多行新增合并
- [x] 重写 `_three_way_merge` 和 `_line_level_merge` (基于 `difflib`)
- [x] 添加合并测试用例 (3个新场景通过)

---

## 🟡 P1 - 中优先级

### 4. 性能趋势分析
- **问题**: 缺乏沿路径的性能趋势数据
- **方案**: `get_performance_trend(node_id) -> List[Dict]`
  - 从 `nodes/{node_id}/logs/` 目录中解析 `result.json` 或 `perf_metrics.json`
- [ ] 实现方法
- [ ] 添加测试

---

## 🟢 P2 - 低优先级

### 6. AST 级别 Diff
### 7. Workspace 恢复优化 (checksum 比对)
### 8. 类型感知的历史压缩策略
### 9. 并发安全 (filelock)

---

## 🔵 P3 - 未来考虑

### 10. 自动清理 / TTL 机制
