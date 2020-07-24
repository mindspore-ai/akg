#!/usr/bin/env python3
# coding: utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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

"""repository"""
__all__ = {
    # sample op
    '2.ZerosLike1.GreaterEqual3.Select123.4': {
        'metadata': {
            'attrs': {
                'enable_double_buffer': False,
            },
        },
        '32_16_28_28_16--': {
            'float32--': {'dim': '0 0 1024 1024'},
        },
    },
    '1.Mul1.Tile1.2': {
        'metadata': {
            'attrs': {
                'enable_variable_mask': True,
            },
        },
    },
    # BNUpdateGrad
    '4.Cast4.ReduceSum1.TensorAdd3.Sqrt1.RealDiv1.Mul7.Cast9.TensorAdd12.Mul14.Mul19.ReduceSum1.514': {
        'metadata': {
            'attrs': {
                'merge_outer_loop_for_multicore': 1,
                'enable_auto_inline': False,
            },
        },
        '32_4_112_112_16-.1_4_1_1_16---': {
            'float16-.float32---': {'dim': '0 0 1 1 0 1 16 16 0 2 1 1 0 3 2 2 0 4 112 112'},
        },
        '32_16_56_56_16-.1_16_1_1_16---': {
            'float16-.float32---': {'dim': '0 0 1 1 0 1 16 16 0 2 1 1 0 3 14 14 0 4 56 56'},
        },
        '32_32_7_7_16-.1_32_1_1_16---': {
            'float16-.float32---': {'dim': '0 0 16 16 0 1 16 16 0 2 1 1 0 3 7 7 0 4 7 7'},
        },
        '32_4_56_56_16-.1_4_1_1_16---': {
            'float16-.float32---': {'dim': '0 0 1 1 0 1 16 16 0 2 1 1 0 3 14 14 0 4 56 56'},
        },
        '32_16_14_14_16-.1_16_1_1_16---': {
            'float16-.float32---': {'dim': '0 0 1 1 0 1 16 16 0 2 4 4 0 3 14 14 0 4 14 14'},
        },
        '32_8_28_28_16-.1_8_1_1_16---': {
            'float16-.float32---': {'dim': '0 0 1 1 0 1 16 16 0 2 1 1 0 3 28 28 0 4 28 28'},
        },
        '32_64_14_14_16-.1_64_1_1_16---': {
            'float16-.float32---': {'dim': '0 0 4 4 0 1 16 16 0 2 1 1 0 3 14 14 0 4 14 14'},
        },
        '32_32_28_28_16-.1_32_1_1_16---': {
            'float16-.float32---': {'dim': '0 0 1 1 0 1 16 16 0 2 1 1 0 3 28 28 0 4 28 28'},
        },
    },
    '4.ReduceSum4.TensorAdd2.Sqrt1.RealDiv1.Mul6.Cast8.TensorAdd12.Mul14.Mul112.ReduceSum1.413': {
        'metadata': {
            'attrs': {
                'merge_outer_loop_for_multicore': 1,
                'enable_auto_inline': False,
            },
        },
        '32_128_7_7_16-.1_128_1_1_16---': {
            'float32.float16.float32---': {'dim': '0 0 16 16 0 1 16 16 0 2 1 1 0 3 7 7 0 4 7 7'},
        },
    },
    # BiasAddGrad
    '1.ReduceSum1.EquivFormat1.Cast1.3': {
        '64_20_16_16.1024': {
            'float16.float32': {'dim': '0 0 1 1 0 1 16 16 0 2 20 20 0 3 16 16'},
        },
        '64_40_16_16.1024': {
            'float16.float32': {'dim': '0 0 1 1 0 1 16 16 0 2 40 40 0 3 16 16'},
        },
        '64_128_16_16.1024': {
            'float16.float32': {'dim': '0 0 1 1 0 1 16 16 0 2 64 64 0 3 16 16'},
        },
        '64_256_16_16.1024': {
            'float16.float32': {'dim': '0 0 1 1 0 1 16 16 0 2 64 64 0 3 16 16'},
        },
        '256_128_16_16.1024': {
            'float16.float32': {'dim': '0 0 1 1 0 1 16 16 0 2 64 64 0 3 16 16'},
        },
        '256_256_16_16.1024': {
            'float16.float32': {'dim': '0 0 1 1 0 1 16 16 0 2 64 64 0 3 16 16'},
        },
    },
    '1.ReduceSum1.Cast1.2': {
        '16_1024.1024': {
            'float16.float32': {'dim': '0 0 32 32 0 1 16 16'},
        },
        '32_1024.1024': {
            'float16.float32': {'dim': '0 0 32 32 0 1 32 32'},
        },
        '4096_1024.1024': {
            'float16.float32': {'dim': '0 0 32 32 0 1 512 512'},
        },
        '2048_1024.1024': {
            'float16.float32': {'dim': '0 0 32 32 0 1 512 512'},
        },
    },
    '2.ReduceSum2.Cast1.Mul13.4': {
        '8192_1024..1024': {
            'float16.float32-': {'dim': '0 0 32 32 0 1 512 512'},
        },
    },
    '2.ReduceSum2.EquivFormat1.Cast1.Mul14.5': {
        '64_1_16_16..1024': {
            'float16.float32-': {'dim': '0 0 1 1 0 1 16 16 0 2 16 16'},
        },
        '64_512_16_16..1024': {
            'float16.float32-': {'dim': '0 0 1 1 0 1 16 16 0 2 64 64 0 3 16 16'},
        },
        '256_512_16_16..4096': {
            'float16.float32-': {'dim': '0 0 1 1 0 1 16 16 0 2 64 64 0 3 16 16'},
        },
    },
    # MaximumGrad
    '2.ZeroLike1.GreaterEqual3.Select123.4': {
        '16_128_1024--': {
            'float32--': {'dim': '0 0 16384 1 0 1 1 1'},
        }
    },
    # MinimumGrad
    '2.ZeroLike1.LessEqual3.Select123.4': {
        '16_128_1024--': {
            'float32--': {'dim': '0 0 16384 1 0 1 1 1'},
        }
    },
    # Fused_LayerNormBetaGamma_11285620133468863670
    '4.Sub34.TensorAdd3.Log1.Mul1.Exp1.Mul15.Mul17.ReduceSum1.ReduceSum9.1112': {
        'metadata': {
            'attrs': {
                'enable_auto_inline': True,
            },
        },
    },
    # Gelu
    '1.Mul11.Mul12.Mul1.TensorAdd14.Mul1.Minimum1.Exp1.Abs3.Mul1.Exp1.TensorAdd1.RealDiv12.Mul16.13': {
        'metadata': {
            'attrs': {
                'pragma_allow_tail_tiling': False,
                'pragma_speedup_tiling': True,
                'pragma_analyze_reuse_buffer': True,
            },
        },
    },
    '1.Cast1.Mul11.Mul12.Mul1.TensorAdd14.Mul1.Minimum1.Exp1.Abs3.Mul1.Exp1.TensorAdd1.RealDiv112.Mul16.Cast1.15': {
        'metadata': {
            'attrs': {
                'pragma_allow_tail_tiling': False,
                'pragma_speedup_tiling': True,
                'pragma_analyze_reuse_buffer': True,
            },
        },
    },
    '3.Mul33.Mul14.Mul1.TensorAdd16.Mul1.Abs1.Mul1.Exp1.TensorAdd1.Minimum5.Exp1.Mul13.Mul12.TensorAdd1.Mul116.Mul18.\
Mul7.Exp1.TensorAdd13.RealDiv18.Mul121.23': {
        'metadata': {
            'attrs': {
                'pragma_allow_tail_tiling': False,
                'pragma_speedup_tiling': True,
                'pragma_analyze_reuse_buffer': True,
            },
        },
    },
    '3.Cast1.Cast4.Mul11.Mul12.Mul1.TensorAdd14.Mul1.Abs1.Mul1.Exp1.TensorAdd1.Minimum5.Exp1.Mul13.Mul12.TensorAdd1.\
Cast18.Mul12.Mul19.Mul8.Exp1.TensorAdd13.RealDiv19.Mul123.Cast1.27': {
        'metadata': {
            'attrs': {
                'pragma_allow_tail_tiling': False,
                'pragma_speedup_tiling': True,
                'pragma_analyze_reuse_buffer': True,
            },
        },
        '256_512_16_16---': {
            'float16---': {'dim': '0 0 4096 4096'},
        },
    },
    # Fused_ReduceSum_Mul
    '2.ReduceSum2.Mul12.3': {
        '1216_30522..30522': {
            'float32--': {
                'metadata': {
                    'attrs': {
                        'pragma_remove_invariant_dependence': True,
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                        'enable_cover_protect_optimize': False,
                    },
                },
                'dim': '0 0 280 1 0 1 57 1'
            },
        },
    },
    # Fused_LambNextMV
    '7.Mul7.Mul7.TensorAdd12.RealDiv18.Sqrt1.TensorAdd1.Mul10.Mul10.TensorAdd12.RealDiv111.RealDiv15.Mul12.TensorAdd9.\
Rsqrt1.Mul15.TensorAdd14.InplaceAssign1819.InplaceAssign11523.1724': {
        'metadata': {
            'enable_double_buffer': True,
        },
        '4096_1024-.1.4096_1024-.1.4096_1024--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'pragma_remove_invariant_dependence': True,
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                        'enable_rewrite_scalar_compute': True,
                    },
                },
                'dim': '0 0 4096 1'
            },
        },
        '1024_4096-.1.1024_4096-.1.1024_4096--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'pragma_remove_invariant_dependence': True,
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                        'enable_rewrite_scalar_compute': True,
                    },
                },
                'dim': '0 0 4096 1'
            },
        },
        '1024_1024-.1.1024_1024-.1.1024_1024--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'pragma_remove_invariant_dependence': True,
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                        'enable_rewrite_scalar_compute': True,
                    },
                },
                'dim': '0 0 4096 1'
            },
        },
        '21128_1024-.1.21128_1024-.1.21128_1024--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'pragma_remove_invariant_dependence': True,
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                        'enable_rewrite_scalar_compute': True,
                    },
                },
                'dim': '0 0 2432 1'
            },
        },
        '30528_1024-.1.30528_1024-.1.30528_1024--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'pragma_remove_invariant_dependence': True,
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                        'enable_rewrite_scalar_compute': True,
                    },
                },
                'dim': '0 0 2048 1'
            },
        },
        '30522-.1.30522-.1.30522--': {
            'float32--------': {'dim': '0 0 2304 1'}
        },
        '30522_1024-.1.30522_1024-.1.30522_1024--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'pragma_remove_invariant_dependence': True,
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                        'enable_rewrite_scalar_compute': True,
                    },
                },
                'dim': '0 0 10174 1'
            },
        },
    },
    # Fused_LambUpdateWithLR
    '6.RealDiv45.Greater7.Select12.Greater8.Select12.Minimum1.Maximum1.Mul110.Mul110.Sub110.InplaceAssign1111.16': {
        'metadata': {
            'attrs': {
                'enable_double_buffer': True,
            },
        },
        '--.1.1024_1024--': {
            'float32------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'enable_mark_multi_core': True,
                        'enable_invariant_hoist': False,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 32 1 0 1 512 1'
            },
        },
        '--.1.1024_4096--': {
            'float32------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'enable_mark_multi_core': True,
                        'enable_invariant_hoist': False,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 64 1'
            },
        },
        '--.1.4096_1024--': {
            'float32------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'enable_mark_multi_core': True,
                        'enable_invariant_hoist': False,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 128 1'
            },
        },
        '--.1.21128_1024--': {
            'float32------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'enable_mark_multi_core': True,
                        'enable_invariant_hoist': False,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 8 1 0 1 512 1'
            },
        },
        '--.1.30528_1024--': {
            'float32------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'enable_mark_multi_core': True,
                        'enable_invariant_hoist': False,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 32 1 0 1 512 1'
            },
        },
        '--.1.30522_1024--': {
            'float32------': {
                'metadata': {
                    'attrs': {
                        'enable_double_buffer': True,
                        'enable_mark_multi_core': True,
                        'enable_invariant_hoist': False,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 6 1 0 1 512 1'
            },
        },
    },
    # Fused_Cast_TransData
    '1.Cast1.TransData1.2': {
        'metadata': {
            'attrs': {
                'enable_auto_inline': True,
            },
        },
        '1024_1024.64_64_16_16': {
            'float32.float16': {
                'dim': '0 0 16 1 0 1 1 1 0 2 16 1 0 3 16 1'
            }
        },
        '1024_4096.256_64_16_16': {
            'float32.float16': {
                'dim': '0 0 64 1 0 1 1 1 0 2 16 1 0 3 16 1'
            }
        },
        '4096_1024.64_256_16_16': {
            'float32.float16': {
                'dim': '0 0 64 1 0 1 1 1 0 2 16 1 0 3 16 1'
            }
        },
        '1216_30522.1908_76_16_16': {
            'float32.float16': {
                'metadata': {
                    'attrs': {
                        'enable_auto_inline': True,
                        'enable_pre_poly_loop_partition': False,
                        'enable_to_three_address': False
                    }
                },
                'dim': '0 0 36 1 0 1 2 1 0 2 16 1'
            }
        },
        '30522_1024.64_1908_16_16': {
            'float32.float16': {
                'metadata': {
                    'attrs': {
                        'enable_auto_inline': True,
                        'merge_outer_loop_for_multicore': 1
                    }
                },
                'dim': '0 0 64 1 0 1 1 1 0 2 16 1 0 3 16 1'
            }
        },
        '1216_1024.64_76_16_16': {
            'float32.float16': {
                'dim': '0 0 16 1 0 1 1 1 0 2 16 1'
            }
        },
    },

    # Fused_Mul_ReduceSum
    '2.Mul12.ReduceSum1.3': {
        '1216_30522-.1216': {
            'float32--': {
                'dim': '0 0 8 1 0 1 2048 1'    
            }
        },        
    },

    #Fused_AddN_ReduceSum_Cast_Mul
    '3.AddN3.ReduceSum1.Cast1.Mul14.36': {
        '8192_1024-..8192_1024.1024': {
            'float16-.float32.float16.float32': {
                'dim': '0 0 64 1 0 1 256 1'    
            }    
        },        
    },

    #Fused_Mul_ReduceSum_Cast_Mul
    '3.Mul23.ReduceSum1.Cast1.Mul14.6': {
        '8192_1024-..1024': {
            'float16-.float32-': {
                'dim': '0 0 64 1 0 1 256 1'    
            }    
        },
    },

    #Fused_ReduceSum_Neg_Mul
    '2.ReduceSum2.Neg1.MUl13.4': {
        '1216_30522.12106-': {
            'float32--': {
                'dim': '0 0 8 1 0 1 2048 1'     
            }    
        },
    },

    # Fused_Reciprocal_ReduceSum_Mul
    '2.Reciprocal2.ReduceSum2.Mul12.24': {
        '.1216_30522.1.30522': {
            'float32---': {
                'metadata': {
                    'attrs': {
                        'pragma_remove_invariant_dependence': True,
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                        'enable_cover_protect_optimize': False,
                    },
                },
                'dim': '0 0 953 1 0 1 16 1'
            },
        },
    },

    # Fused_Mul_Mul_TensorAdd
    '2.Mul1.Mul3.TensorAdd12.34': {
        '16_1_1_512.16_16_512_512.16_1_1_512.16_16_512_512': {
            'float16---': {'dim': '0 0 2 1 0 1 128 1 0 2 2 1 0 3 32 1'},
        },
    },

    # Fused_Cast_Cast_Mul_TensorAdd
    '3.Cast2.Cast4.Mul13.TensorAdd13.46': {
        '1024-.8192_1024.1024.8192_1024': {
            'float32-.float16--': {
                'metadata': {
                    'attrs': {
                        'pragma_set_all_coincident': True,
                        'enable_post_poly_loop_partition': False,
                        'multicore_loop_switch_hoist': False,
                    },
                },
                'dim': '0 0 1024 1 0 1 32 1'
            },
        },
    },

    # Fused_Mul_TensorAdd
    '2.Mul2.TensorAdd12.3': {
        '30522_1024--': {
            'float32--': {'dim': '0 0 13376 1'},
        },
        '512_1024--': {
            'float32--': {'dim': '0 0 13376 1'},
        },
        '1024_1024--': {
            'float32--': {'dim': '0 0 13376 1'},
        },
        '4096_1024--': {
            'float32--': {'dim': '0 0 13376 1'},
        },
        '1024_4096--': {
            'float32--': {'dim': '0 0 13376 1'},
        },
    },

    # Fused_LambNextMV
    '7.Mul7.Mul7.TensorAdd12.RealDiv18.Sqrt1.TensorAdd1.Mul10.Mul10.TensorAdd12.RealDiv111.RealDiv15.Mul12.\
TensorAdd12.InplaceAssign1516.InplaceAssign11220.1721': {
        '30522_1024-.1.30522_1024-.1.30522_1024--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_rewrite_scalar_compute': True,
                        'pragma_remove_invariant_dependence': True,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 10174 1'
            },
        },
        '1024_4096-.1.1024_4096-.1.1024_4096--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_rewrite_scalar_compute': True,
                        'pragma_remove_invariant_dependence': True,
                        'multicore_scalar_rearrange': True,
                        'pragma_speedup_tiling': True,
                        'pragma_analyze_reuse_buffer': True,
                    },
                },
            },
        },
        '1024_1024-.1.1024_1024-.1.1024_1024--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_rewrite_scalar_compute': True,
                        'pragma_remove_invariant_dependence': True,
                        'multicore_scalar_rearrange': True,
                        'pragma_speedup_tiling': True,
                        'pragma_analyze_reuse_buffer': True,
                    },
                },
            },
        },
        '30522-.1.30522-.1.30522--': {
            'float32--------': {'dim': '0 0 2304 1'},
        },
        '4096_1024-.1.4096_1024-.1.4096_1024--': {
            'float32--------': {
                'metadata': {
                    'attrs': {
                        'enable_rewrite_scalar_compute': True,
                        'pragma_remove_invariant_dependence': True,
                        'multicore_scalar_rearrange': True,
                        'pragma_speedup_tiling': True,
                        'pragma_analyze_reuse_buffer': True,
                    },
                },
            },
        },
    },

    # Fused_LambUpdateWithLR
    '5.RealDiv45.Minimum1.Maximum1.Mul16.Mul16.Sub16.InplaceAssign117.11': {
        '-.1.30522_1024--': {
            'float32-----': {
                'metadata': {
                    'attrs': {
                        'pragma_remove_invariant_dependence': True,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 6 1 0 1 1024 1'
            },
        },
        '-.1.4096_1024--': {
            'float32-----': {
                'metadata': {
                    'attrs': {
                        'enable_mark_multi_core': True,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 128 1'
            },
        },
        '-.1.1024_4096--': {
            'float32-----': {
                'metadata': {
                    'attrs': {
                        'pragma_remove_invariant_dependence': True,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 4 1 0 1 4096 1'
            },
        },
        '-.1.1024_1024--': {
            'float32-----': {
                'metadata': {
                    'attrs': {
                        'pragma_remove_invariant_dependence': True,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 16 1 0 1 1024 1'
            },
        },
        '-.1.512_1024--': {
            'float32-----': {
                'metadata': {
                    'attrs': {
                        'pragma_remove_invariant_dependence': True,
                        'multicore_scalar_rearrange': True,
                    },
                },
                'dim': '0 0 16 1 0 1 1024 1'
            },
        },
    },

    # Mul_TensorAdd
    '2.EquivFormat.Mul13.EquivFormat3.TensorAdd12.5': {
        '16_16_32_32_16_16.16_1_1_512.16_16_32_32_16_16': {
            'float16--': {
                'metadata': {
                    'attrs': {
                        'enable_mark_multi_core': True,
                        'multicore_loop_switch_hoist': False,
                        'multicore_scalar_rearrange': True,
                        'enable_post_poly_loop_partition': False,
                    },
                },
                'dim': '0 0 1 1 0 1 2 1 0 2 16 1 0 3 1 1 0 4 32 1 0 5 16 1'
            },
        },
    },
}
