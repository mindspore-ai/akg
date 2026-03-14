"""评分逻辑模块。

评分规则:
    - 正确性是前提，不正确的 case 得 0 分
    - 性能分 = f(speedup), 满分 100
    - 不同 tier 有不同的权重系数（难度越高权重越大）
"""

TIER_WEIGHTS: dict[str, float] = {
    "t1": 1.0,
    "t2": 1.5,
    "t3": 2.0,
    "t4": 2.5,
    "t5": 3.0,
}

DEFAULT_TIER_WEIGHT = 1.0


def get_tier_weight(tier: str) -> float:
    return TIER_WEIGHTS.get(tier, DEFAULT_TIER_WEIGHT)


def compute_case_score(speedup: float) -> float:
    """根据 speedup 计算单个 case 的原始分 (0~100)。

    - speedup < 1.0: 比 baseline 慢，线性折扣 [0, 60)
    - speedup == 1.0: 基础分 60
    - speedup > 1.0: 从 60 向 100 递增，speedup >= 5.0 时封顶 100
    """
    if speedup <= 0:
        return 0.0
    if speedup < 1.0:
        return 60.0 * speedup
    bonus = min(speedup - 1.0, 4.0) / 4.0 * 40.0
    return 60.0 + bonus


def compute_weighted_score(tier: str, speedup: float) -> float:
    """计算加权后的 case 得分。"""
    raw = compute_case_score(speedup)
    return raw * get_tier_weight(tier)
