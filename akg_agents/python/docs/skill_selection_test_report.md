# Skill 选择测试报告

> DSL: `triton_ascend` | Framework: `torch` | 加载: 33 skills

## Category 分布

| Layer | Category | Count |
|-------|----------|-------|
| 2 | case | 21 |
| 1 | example | 2 |
| 0 | fundamental | 6 |
| 1 | guide | 4 |

## Guide Skills 关键词

| Skill | operator_patterns | algorithms |
|-------|-------------------|------------|
| triton-ascend-attention | attention | self-attention, cross-attention, flash-attention, scaled-dot-product-attention |
| triton-ascend-elementwise | elementwise | add, mul, relu, sigmoid, tanh, gelu, exp, log, div, sub, sqrt, pow |
| triton-ascend-matmul | matmul | matmul, bmm, linear |
| triton-ascend-reduce | reduce | sum, mean, max, min, softmax, layernorm, logsoftmax |

---

## 测试结果明细

### relu (elementwise)

- **op_name**: `relu`
- **source**: `KernelBench/level1/19_ReLU.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['relu']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 8 skills, ~15242 chars (~5080 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['relu']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~42038 chars (~14012 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['relu']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~42038 chars (~14012 tokens)

---

### gelu (elementwise)

- **op_name**: `gelu`
- **source**: `bench_lite/t1/gelu.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'gelu', 'sqrt']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['linear']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 9 skills, ~18324 chars (~6108 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'gelu', 'sqrt']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['linear']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~45120 chars (~15040 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'gelu', 'sqrt']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['linear']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~45120 chars (~15040 tokens)

---

### softmax (reduce)

- **op_name**: `softmax`
- **source**: `bench_lite/t1/softmax.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'exp']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['sum', 'max', 'softmax']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 9 skills, ~16054 chars (~5351 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'exp']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['sum', 'max', 'softmax']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~42850 chars (~14283 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'exp']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['sum', 'max', 'softmax']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~42850 chars (~14283 tokens)

---

### matmul_basic (matmul)

- **op_name**: `matmul`
- **source**: `bench_lite/t1/matmul_basic.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 9 skills, ~18324 chars (~6108 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~45120 chars (~15040 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~45120 chars (~15040 tokens)

---

### matmul_biasadd (matmul+elementwise)

- **op_name**: `matmul_biasadd`
- **source**: `bench_lite/t1/matmul_biasadd.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['add', 'mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 9 skills, ~18324 chars (~6108 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['add', 'mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~45120 chars (~15040 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['add', 'mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~45120 chars (~15040 tokens)

---

### matmul_large_k (matmul)

- **op_name**: `matmul_large_k`
- **source**: `KernelBench/level1/6_Matmul_with_large_K.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 9 skills, ~18324 chars (~6108 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~45120 chars (~15040 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['matmul', 'matmul']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~45120 chars (~15040 tokens)

---

### scaled_dot_product_attention (attention)

- **op_name**: `scaled_dot_product_attention`
- **source**: `EvoKernel/level1/97_ScaledDotProductAttention.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-attention (1492 chars) ← 关键词: `['attention']`

**L1 guide 排除** (3):
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 8 skills, ~14350 chars (~4783 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-attention (1492 chars) ← 关键词: `['attention']`

**L1 guide 排除** (3):
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~41146 chars (~13715 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-attention (1492 chars) ← 关键词: `['attention']`

**L1 guide 排除** (3):
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~41146 chars (~13715 tokens)

---

### max_over_dim (reduce)

- **op_name**: `max_reduction`
- **source**: `KernelBench/level1/49_Max_reduction_over_a_dimension.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['max']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 8 skills, ~13670 chars (~4556 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['max']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~40466 chars (~13488 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['max']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~40466 chars (~13488 tokens)

---

### sum_reduction (reduce)

- **op_name**: `sum_reduction`
- **source**: `KernelBench/level1/47_Sum_reduction.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['reduce', 'sum']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 8 skills, ~13670 chars (~4556 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['reduce', 'sum']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~40466 chars (~13488 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['reduce', 'sum']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-elementwise (2384 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~40466 chars (~13488 tokens)

---

### cumprod (scan)

- **op_name**: `cumprod`
- **source**: `KernelBench/level1/90_cumprod.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 8 skills, ~15242 chars (~5080 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~42038 chars (~14012 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~42038 chars (~14012 tokens)

---

### layernorm_gated (reduce+elementwise)

- **op_name**: `layernorm_gated`
- **source**: `bench_lite/t3/layernorm_gated.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (4):
- ✅ triton-ascend-attention (1492 chars) ← 关键词: `['attention']`
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'sigmoid', 'sqrt', 'pow']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['linear']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['mean', 'layernorm']`

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 11 skills, ~20628 chars (~6876 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (4):
- ✅ triton-ascend-attention (1492 chars) ← 关键词: `['attention']`
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'sigmoid', 'sqrt', 'pow']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['linear']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['mean', 'layernorm']`

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 32 skills, ~47424 chars (~15808 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (4):
- ✅ triton-ascend-attention (1492 chars) ← 关键词: `['attention']`
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul', 'sigmoid', 'sqrt', 'pow']`
- ✅ triton-ascend-matmul (3082 chars) ← 关键词: `['linear']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['mean', 'layernorm']`

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 32 skills, ~47424 chars (~15808 tokens)

---

### fused_silu_and_mul (elementwise/SwiGLU)

- **op_name**: `fused_silu_mul`
- **source**: `bench_lite/t1/fused_silu_and_mul.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 8 skills, ~15242 chars (~5080 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~42038 chars (~14012 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~42038 chars (~14012 tokens)

---

### add_rmsnorm_cast (fused reduce+elementwise)

- **op_name**: `add_rmsnorm_cast`
- **source**: `bench_lite/t2/add_rmsnorm_cast.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['add', 'sqrt', 'pow']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['mean']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 9 skills, ~16054 chars (~5351 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['add', 'sqrt', 'pow']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['mean']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~42850 chars (~14283 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['add', 'sqrt', 'pow']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['mean']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~42850 chars (~14283 tokens)

---

### rope (position encoding)

- **op_name**: `rope`
- **source**: `bench_lite/t2/rope.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 8 skills, ~15242 chars (~5080 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~42038 chars (~14012 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (1):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['mul']`

**L1 guide 排除** (3):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)
- ❌ triton-ascend-reduce (812 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 29 skills, ~42038 chars (~14012 tokens)

---

### sigmoid_scale_sum (elementwise+reduce)

- **op_name**: `sigmoid_scale_sum`
- **source**: `bench_lite/t1/sigmoid_scale_sum.py`

#### stage = initial

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['sigmoid']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['sum']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**总计**: 9 skills, ~16054 chars (~5351 tokens)

---

#### stage = debug

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['sigmoid']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['sum']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~42850 chars (~14283 tokens)

---

#### stage = optimize

**L0 固定注入** (6):
- triton-ascend-api (1709 chars)
- triton-ascend-optimization (1680 chars)
- triton-ascend-debugging (1874 chars)
- triton-ascend-grid-config (1997 chars)
- triton-ascend-memory (1408 chars)
- triton-ascend-basics (1656 chars)

**L1 guide 命中** (2):
- ✅ triton-ascend-elementwise (2384 chars) ← 关键词: `['sigmoid']`
- ✅ triton-ascend-reduce (812 chars) ← 关键词: `['sum']`

**L1 guide 排除** (2):
- ❌ triton-ascend-attention (1492 chars)
- ❌ triton-ascend-matmul (3082 chars)

**L1 example** (1): ['triton-ascend-examples-torch']

**L2 case 候选池** (21):
- triton-ascend-case-reduction-amax-medium
- triton-ascend-case-reduction-amin-large
- triton-ascend-case-reduction-amin-atomic
- triton-ascend-case-reduction-amin-medium
- triton-ascend-case-reduction-mean-large
- ... 共 21 个

**总计**: 30 skills, ~42850 chars (~14283 tokens)

---


---

## 汇总表（initial 阶段）

| 测试用例 | guide 命中 | guide 排除 | 命中原因 | ~tokens |
|---------|-----------|-----------|---------|---------|
| relu (elementwise) | elementwise | attention, matmul, reduce | triton-ascend-elementwise←`['relu']` | 5080 |
| gelu (elementwise) | elementwise, matmul | attention, reduce | triton-ascend-elementwise←`['mul', 'gelu', 'sqrt']`; triton- | 6108 |
| softmax (reduce) | elementwise, reduce | attention, matmul | triton-ascend-elementwise←`['mul', 'exp']`; triton-ascend-re | 5351 |
| matmul_basic (matmul) | elementwise, matmul | attention, reduce | triton-ascend-elementwise←`['mul']`; triton-ascend-matmul←`[ | 6108 |
| matmul_biasadd (matmul+elementwise) | elementwise, matmul | attention, reduce | triton-ascend-elementwise←`['add', 'mul']`; triton-ascend-ma | 6108 |
| matmul_large_k (matmul) | elementwise, matmul | attention, reduce | triton-ascend-elementwise←`['mul']`; triton-ascend-matmul←`[ | 6108 |
| scaled_dot_product_attention (attention) | attention | elementwise, matmul, reduce | triton-ascend-attention←`['attention']` | 4783 |
| max_over_dim (reduce) | reduce | attention, elementwise, matmul | triton-ascend-reduce←`['max']` | 4556 |
| sum_reduction (reduce) | reduce | attention, elementwise, matmul | triton-ascend-reduce←`['reduce', 'sum']` | 4556 |
| cumprod (scan) | elementwise | attention, matmul, reduce | triton-ascend-elementwise←`['mul']` | 5080 |
| layernorm_gated (reduce+elementwise) | attention, elementwise, matmul, reduce |  | triton-ascend-attention←`['attention']`; triton-ascend-eleme | 6876 |
| fused_silu_and_mul (elementwise/SwiGLU) | elementwise | attention, matmul, reduce | triton-ascend-elementwise←`['mul']` | 5080 |
| add_rmsnorm_cast (fused reduce+elementwise) | elementwise, reduce | attention, matmul | triton-ascend-elementwise←`['add', 'sqrt', 'pow']`; triton-a | 5351 |
| rope (position encoding) | elementwise | attention, matmul, reduce | triton-ascend-elementwise←`['mul']` | 5080 |
| sigmoid_scale_sum (elementwise+reduce) | elementwise, reduce | attention, matmul | triton-ascend-elementwise←`['sigmoid']`; triton-ascend-reduc | 5351 |