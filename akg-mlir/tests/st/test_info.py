# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AKG-MLIR st test."""
import os
import pytest

def run_case(info, target='ascend'):
    """ run a test case """
    pwd = os.path.dirname(os.path.abspath(__file__))
    info = os.path.join(pwd, target, info)
    cmd = "akg_benchmark -e " +  target + " -f " + info
    ret = os.system(cmd)
    assert ret == 0

# ── CogVideoX 融合算子 ST 用例 (7 类 × 2 = 14 个) ──────────────

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_sub_add():
    """
    Feature: AKG compile test.
    Description: fuse sub and add.
    Expectation: success
     """
    run_case("fused_sub_add.info")

# ── 1. 激活/条件运算 (Mul_Maximum_Select, Mul_RealDiv_Exp scalar broadcast) ──

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_mul_maximum_select():
    """Description: Mul→Maximum→Select→Mul→Select→Assign, ReLU/clamp-like. Expectation: success"""
    run_case("Fused_Mul_Maximum_Select_Mul_Select_Assign_fusion_6977071865884904062.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_mul_realdiv_exp():
    """Description: Mul→RealDiv→Exp, scalar broadcast [1536]vs[1], float16. Expectation: success"""
    run_case("Fused_Mul_RealDiv_Exp_fusion_2550894545544229730.info")

# ── 2. 梯度/累加 (AssignAdd, Add) ──

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_assignadd():
    """Description: AssignAdd gradient accumulation. Expectation: success"""
    run_case("Fused_AssignAdd_3064818203767919110.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_add():
    """Description: Add elementwise. Expectation: success"""
    run_case("Fused_Add_fusion_15759721826023052688.info")

# ── 3. 归一化/除法 (Sub→RealDiv, Reciprocal→Cast) ──

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_sub_realdiv():
    """Description: Sub→RealDiv normalization. Expectation: success"""
    run_case("Fused_Sub_RealDiv_fusion_16158667714098616853.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_div_reshape_2():
    """Description: Div→Reshape normalization variant 2. Expectation: success"""
    run_case("Fused_Div_Reshape_split_1859509568641689366.info")

# ── 4. 精度转换链 (Cast→Mul 2D broadcast, Cast→Sub) ──

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_cast_mul():
    """Description: Cast→Mul, 2D broadcast [1,1]vs[1,1536]. Expectation: success"""
    run_case("Fused_Cast_Mul_fusion_11325474031898152154.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_cast_sub():
    """Description: Cast→Sub precision conversion. Expectation: success"""
    run_case("Fused_Cast_Sub_fusion_8883087981125364622.info")

# ── 5. 归约运算 (ReduceX axis=0 + ReduceY axis=1 + AllReduce axis=[0,1]) ──

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_mul_reducesum_x():
    """Description: Mul→ReduceSum axis=0 (ReduceX), input [3072]→scalar,
    scalar broadcast [3072]vs[1]. Expectation: success"""
    run_case("Fused_Mul_ReduceSum_split_1421551648179530053.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_mul_reducesum_all():
    """Description: Mul→ReduceSum axis=[0,1] (AllReduce),
    input [768,512]→scalar, scalar broadcast [768,512]vs[1]. Expectation: success"""
    run_case("Fused_Mul_ReduceSum_split_13305325157517622155.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_reducesum_y_large():
    """Description: ReduceSum axis=1 (ReduceY), input [1,5100,3072]→[1,1,3072] keep_dims. Expectation: success"""
    run_case("Fused_ReduceSum_split_10316266540114966563.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_reducesum_y():
    """Description: ReduceSum axis=1 (ReduceY), input [1,28,3072]→[1,1,3072] keep_dims. Expectation: success"""
    run_case("Fused_ReduceSum_split_9250037208464391444.info")

# ── 6. 逐元素算术链 (Tile→Div BroadcastTo, Div→Reshape) ──

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_tile_div():
    """Description: Tile(BroadcastTo)→Div, broadcast [1,1]→[1,5222400] then scalar div. Expectation: success"""
    run_case("Fused_Tile_Div_fusion_18189358198697709381.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_div_reshape():
    """Description: Div→Reshape. Expectation: success"""
    run_case("Fused_Div_Reshape_split_18352953218106345430.info")

# ── 7. 累加/聚合 (Add×17 more_split, LessEqual) ──

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_add_chain_more_split():
    """Description: Add×17 chain more_split, 24 scalar inputs accumulation. Expectation: success"""
    run_case("Fused_AddChain_more_split_9542740108961475159.info")

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_fuse_lessequal():
    """Description: LessEqual comparison. Expectation: success"""
    run_case("Fused_LessEqual_fusion_1678676461603857418.info")
