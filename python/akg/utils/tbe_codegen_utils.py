#!/usr/bin/env python3
# coding: utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import logging

logging.getLogger().setLevel(logging.INFO)


def copy_to_akg_kernel_meta(kernel_name, postfixs):
    akg_kernel_mate_str = "akg_kernel_meta"
    source = os.path.realpath(os.getenv('MS_COMPILER_CACHE_PATH', './'))
    import shutil
    target = source + "/" + akg_kernel_mate_str + "/" + kernel_name
    source = source + "/" + "kernel_meta/" + kernel_name
    if not os.path.exists(akg_kernel_mate_str):
        os.mkdir(akg_kernel_mate_str)
    for postfix in postfixs:
        if os.path.exists(source + postfix):
            try:
                shutil.move(source + postfix, target + postfix)
            except IOError as e:
                logging.error("Unable to move file. {}".format(e))
            except Exception as e:
                logging.error("Unexpected error:", e)
        else:
            logging.info("Move {} fail, exit.".format(source + postfix))
            return False
    return True


def clean_env():
    import gc
    import sys

    imported_modules = set(sys.modules.keys())
    for obj_key in imported_modules:
        if "conda" in obj_key:
            continue
        if "akg" in obj_key or "topi" in obj_key or "tvm" in obj_key:
            del sys.modules[obj_key]
            try:
                del globals()[obj_key]
            except KeyError:
                pass
            try:
                del locals()[obj_key]
            except KeyError:
                pass

    gc.collect()


def auto_init_soc(ascend_type):
    from tbe.common.platform import set_current_compile_soc_info
    set_current_compile_soc_info(ascend_type)


def build_npu_for_akg(kernel_name,
                      stmt=None,
                      arg_list=None,
                      is_dynamic=False,
                      cfg=None,
                      simple_mode=False):
    import tbe
    from tbe.tvm.tir import transform
    from tbe.tvm.driver.cce_build_module import _count_time, generate_cce_code
    from tbe.common.buildcfg import set_current_build_config
    from tbe.common.buildcfg.buildcfg_mapping import dynamic_shape, disable_vectorize, tik, enable_const_fold, \
        dynamic_tik, instrument_bound_checkers, tbe_workspace_size_list_length
    
    set_current_build_config(tbe_workspace_size_list_length,
                             tbe.tvm.runtime.cce_runtime.tbe_workspace_size_list_length())

    if stmt is None or arg_list is None:
        raise ValueError("No json, exit.")

    func = tbe.tvm.tir.PrimFunc(arg_list, stmt)
    mod = tbe.tvm.IRModule({kernel_name : func})
    # _static_lower_phase_0
    mod = transform.InjectSocVersion()(mod)
    mod = transform.DeduceOpPlatform()(mod)
    mod = transform.EmitInsn()(mod)

    # phase 1 _static_lower_phase_emit_insn
    mod = transform.InjectMultiCoreSync()(mod)
    mod = transform.SplitCoproc()(mod)
    mod = transform.SequenceSprInsn()(mod)

    # phase 2
    mod = transform.TikDoubleBufferSupport()(mod)
    mod = transform.InjectPipeBuffer()(mod)
    mod = transform.OptimizeDMA()(mod)
    mod = transform.SubstituteInstr()(mod)
    mod = transform.InjectAccessPtrMSG()(mod)
    mod = transform.InjectPipe()(mod)
    mod = transform.DeSequenceSprInsn()(mod)
    mod = transform.CanonicalSimplify()(mod)
    mod = transform.SetSPROptimizer()(mod)
    if cfg[enable_const_fold]:
        mod = transform.ConstantFolding()(mod)
    if not simple_mode:
        mod = transform.LoopPartition()(mod)
    if cfg[disable_vectorize]:
        mod = transform.SkipVectorize()(mod)
    else:
        mod = transform.VectorizeLoop()(mod)
    mod = transform.InjectVirtualThread()(mod)

    # phase 3 _static_lower_phase_3
    mod = transform.StorageRewriteCCE()(mod)
    mod = transform.ReorderProcess()(mod)
    if cfg[tik] and cfg[dynamic_tik]:
        mod = transform.TikDynamicShapeAllocMem()(mod)
    mod = transform.UnrollLoop()(mod)

    mod = transform.AutoFuseBuffer()(mod)
    mod = transform.SetCacheMode()(mod)
    mod = transform.Simplify()(mod)
    mod = transform.GMConflictElimination()(mod)
    mod = transform.MarkScalarCoreType()(mod)

    # phase 4 _static_lower_phase_4
    mod = transform.JumpInstructionElimination()(mod)
    mod = transform.InjectSync()(mod)
    mod = transform.PackIntrinArgConfig()(mod)
    mod = transform.RemoveAccessPtrMSG()(mod)
    mod = transform.Simplify()(mod)
    mod = transform.GmAddrPrompt()(mod)
    mod = transform.InsertCheckInvalidAccessOfDDR()(mod)
    mod = transform.RemoveNoOp()(mod)
    mod = transform.DeviceMark()(mod)
    if cfg[instrument_bound_checkers]:
        mod = transform.InstrumentBoundCheckers()(mod)
    mod = transform.ConvertFloorDivToTruncDiv()(mod)
    mod = transform.BuildVirtualCore()(mod)

    _count_time(mod)
    mod = transform.SplitCoreCode()(mod)
    generate_cce_code(mod, "cce", None)


def build_tbe_codegen(kernel_name, stmt_json, arg_json, ascend_type=None, is_dynamic=False):
    import sys
    copy_modules = sys.modules.copy()
    clean_env()

    print("build_cce_for_akg")
    import tbe
    from tbe.common.buildcfg.default_buildcfg import cce_default_static_build_config
    from tbe.common.buildcfg.ascend import AscendPassContext
    from tbe.common.buildcfg.buildcfg_mapping import dump_cce_code, save_temp_cce_file, disable_vectorize, \
        instrument_bound_checkers, partition_const_loop, auto_unroll_max_step, auto_unroll_max_depth, \
        auto_unroll_max_extent, unroll_explicit, dynamic_shape, enable_multicore_sync_with_atomic, \
        kernel_meta_parent_dir
    cfg = cce_default_static_build_config.copy()
    cfg[dump_cce_code] = False
    cfg[save_temp_cce_file] = True
    cfg[disable_vectorize] = False
    cfg[instrument_bound_checkers] = False
    cfg[partition_const_loop] = False
    cfg[auto_unroll_max_step] = 0
    cfg[auto_unroll_max_depth] = 8
    cfg[auto_unroll_max_extent] = 0
    cfg[unroll_explicit] = True
    cfg[dynamic_shape] = False
    cfg[enable_multicore_sync_with_atomic] = True
    cfg[kernel_meta_parent_dir] = os.path.realpath(os.getenv('MS_COMPILER_CACHE_PATH', './'))
    if ascend_type is None:
        ascend_type = "Ascend910"
    auto_init_soc(ascend_type)

    stmt = tbe.tvm.ir.load_json(stmt_json)
    arg_list = []
    for buff in arg_json:
        arg_list.append(tbe.tvm.ir.load_json(buff))
    with AscendPassContext(config=cfg):
        build_npu_for_akg(kernel_name,
                          stmt,
                          arg_list,
                          is_dynamic=is_dynamic,
                          cfg=cfg)
    postfixs = [".o", ".cce", ".json"]
    is_success = copy_to_akg_kernel_meta(kernel_name, postfixs)
    sys.modules = copy_modules
    return is_success
