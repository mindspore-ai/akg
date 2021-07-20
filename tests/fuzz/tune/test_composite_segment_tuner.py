# Copyright 2021 Huawei Technologies Co., Ltd
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

"""test composite json tuning"""
import os
import sys
import time
import getopt
import logging
import json
import math
import traceback
import multiprocessing
import subprocess
from functools import reduce
from akg import composite
from akg.composite.build_module import generate_trait
from akg.composite.build_module import parallel_json_split, stitch_json_split
from tests.prev_version_auto_tune.job import launch_json
from tests.prev_version_auto_tune.runner import get_attr_from_config
from akg.utils import kernel_exec as utils
import akg.composite.peel as pt
from tests.common.gen_json_data import gen_json_data
from tests.st.composite.test_composite_json import _compare_func

logging.getLogger().setLevel(logging.INFO)


def print_usage():
    logging.warning("Usage: python test_composite_tune.py <kernel_meta_path or info_file> [-f] -o <repo.json>")
    logging.warning("\nOptions:")
    logging.warning("<kernel_meta_path or info_file>")
    logging.warning("  kernel_meta directory or single file name, '*.info' or '*.json'.")
    logging.warning("-f")
    logging.warning("  tune for all jsons, including those already in repository")
    logging.warning("-o <repo.json>")
    logging.warning("  relative path of output repository")


class ProfilingDirCleaner(object):
    def __init__(self):
        self.profiling_dir = None
        profiling_dir = str(os.environ['PROFILING_DIR'])
        if os.path.isdir(profiling_dir):
            self.profiling_dir = profiling_dir

    def __enter__(self):
        if self.profiling_dir is not None:
            subprocess.run("rm -rf %s/JOB*" % self.profiling_dir, shell=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiling_dir is not None:
            subprocess.run("rm -rf %s/JOB*" % self.profiling_dir, shell=True)


def tuning_with_1block(json_str, tune_level=0, repo_path="repo.json", skip_exist=False):
    json_obj = json.loads(json_str)
    json_obj["extra"] = {"BlockMode": "single_block"}
    mod_json_str = json.dumps(json_obj)

    iter_times = [80, 160, 320] if tune_level == 0 else [15, 15, 15]
    self_attrs = None
    tuning_attrs = ['enable_pre_poly_loop_partition',
                    'enable_post_poly_loop_partition']

    best_results = launch_json(debug_mode=False, save_res=True, input_str=mod_json_str,
                               repo_path=repo_path, all_space=False, skip_exist=skip_exist,
                               skip_file=False, extra_tune=False, self_attrs=None,
                               tuning_attrs=tuning_attrs, iter_times=iter_times)

    attrs = {}
    best_time = float("inf")
    if best_results is not None and len(best_results) == 1:
        bst = best_results[0]
        if bst is not None:
            if isinstance(bst, dict):
                # attrs = bst.get("metadata", {}).get("attrs", {})
                raise RuntimeError("Unsupported now...")
            else:
                best_time = bst.best_time
                index_table = bst.index_table
                if bst.best_config is not None:
                    best_config = bst.best_config.input
                    attrs = get_attr_from_config(best_config, index_table)

    tuned_info = {"elapse": best_time, "args": attrs}
    return tuned_info


def get_profiling(desc: str, results: dict, attrs=None):
    mod = composite.build(desc, attrs, poly=True)
    inputs, expect, output_indexes = gen_json_data(desc)
    output, stat_info = utils.mod_launch(mod, list(inputs), output_indexes, device_id=utils.get_device_id())

    results["run_time"] = stat_info["run_time"]
    if not all(map(_compare_func, output if isinstance(output, (list, tuple)) else [output],
                   expect if isinstance(expect, (list, tuple)) else [expect])):
        results["cmp"] = False
    else:
        results["cmp"] = True


def get_repo(repo, keys, default=None):
    for key in keys:
        repo = repo.get(key)
        if not repo:
            return default
    return repo


def record_parallel_repo(desc, repo_path, args_and_plan):
    desc_json = json.loads(desc)
    compute, shape, dtype = generate_trait(desc_json)
    should_export = True
    repo = {}
    if os.path.isfile(repo_path):
        with open(repo_path, 'r') as f:
            json_str = f.read()
            json_str = "{}" if json_str == "" else json_str
            repo = json.loads(json_str)
            if get_repo(repo, [compute, shape, dtype]):
                should_export = False
    if not should_export:
        logging.warning("{} already exist in repo!".format(desc_json["op"]))
        return

    with open(repo_path, "w") as f:
        write_obj = {}
        plans = [
            {
                "attrs": {
                    "block_plan": plan[0],
                    "peeling": plan[1],
                    **args["args"]
                },
                "best_cycles": args["elapse"]
            } for args, plan in args_and_plan]
        write_obj[compute] = {
            shape: {
                dtype: {
                    "BlockPlan": plans,
                    "file_name": desc_json["op"]
                }
            }
        }
        repo.update(write_obj)
        f.write(json.dumps(repo, indent=4))
    logging.info("Success! Record done!")
    return


def record_stitch_repo(desc, repo_path, tuning_info):
    desc_dict = json.loads(desc)
    compute, shape, dtype = generate_trait(desc_dict)
    should_export = True
    repo = {}
    if os.path.isfile(repo_path):
        with open(repo_path, 'r') as f:
            json_str = f.read()
            json_str = "{}" if json_str == "" else json_str
            repo = json.loads(json_str)
            if get_repo(repo, [compute, shape, dtype]) is not None:
                should_export = False
    if not should_export:
        logging.warning("{} already exist in repo!".format(desc_dict["op"]))
        return
    with open(repo_path, 'w') as f:
        write_obj = {
            compute: {
                shape: {
                    dtype: {
                    }
                }
            }
        }
        for k, v in tuning_info["attrs"].items():
            write_obj[compute][shape][dtype][k] = v
        write_obj[compute][shape][dtype]["best_cycles"] = tuning_info["run_time"]
        write_obj[compute][shape][dtype]["file_name"] = desc_dict["op"]
        repo.update(write_obj)
        f.write(json.dumps(repo, indent=4))
        logging.info("Success! Record done!")


def plan_block(candidate: list):
    def _get_blocks_by_candidate(block_candidate, candidate):
        def _judge_divie(block, axis, peel_spaces):
            for space in peel_spaces:
                # Only try one axis peel
                if len(space) == 1:
                    axis_tmp, tile_tmp = space[0]
                    if axis_tmp == axis and tile_tmp == block:
                        return True
            return False

        plans = []
        for bc, [peel_info, spaces, _] in zip(block_candidate, candidate):
            bc = bc if bc > 0 else 1
            hit = False
            peel_axis, _ = list(peel_info.items())[0] # only one outer axis
            block_hint = bc
            for i in range(bc):
                # spaces: [[[0, 20], [2, 16]], [[0, 10], [2, 16]], ...]
                hit = _judge_divie(bc - i, peel_axis, spaces)
                if hit:
                    block_hint = bc - i
                    break
            if not hit:
                raise RuntimeError("Cannot make block plan!")
            peel_config = "{} {}".format(peel_axis, peel_info[peel_axis])
            plans.append((block_hint, peel_config))
        # plans: [(16, "0 16"), (8, "2 8"), ...]
        return plans

    def _calculate_cost(peel):
        cost = 1
        for _, dim in peel.items():
            cost *= dim
        return cost

    BLOCK_TOTAL = 32
    costs = [_calculate_cost(peel) * info["elapse"] for peel, _, info in candidate]
    min_cost = min(costs)
    coefs = [cost / min_cost for cost in costs]
    core_base = BLOCK_TOTAL / sum(coefs)
    block_candidate = [math.floor(coef * core_base) for coef in coefs]
    return _get_blocks_by_candidate(block_candidate, candidate)

def tune_parallel_segment(desc_in: str, repo_path: str):
    def _get_max_peel(peel):
        # only outer peel axis now.
        peel_space_strs = peel.get_peeling_space()
        res = {}
        peel_spaces = []
        outer_axis = 1 << 64
        for peel_str in peel_space_strs:
            peel_list = [int(s) for s in peel_str.split(" ")]
            peel_space = [peel_list[i:i + 2] for i in range(0, len(peel_list), 2)]
            for axis, tile in peel_space:
                # find maximum size for each axis
                if tile > res.get(axis, 0):
                    res.update({axis: tile})
                if axis < outer_axis:
                    outer_axis = axis
            peel_spaces.append(peel_space)
        return {outer_axis: res[outer_axis]}, "{} {}".format(outer_axis, res[outer_axis]), peel_spaces

    def _tune(body_desc):
        best_tuned_info = tuning_with_1block(body_desc)  # here should return elapse time and tiling ars
        if best_tuned_info.get("elapse", float("inf")) == float("inf"):
            print("Cannot get tuning info, will check body's elapse by profiling!")
            with ProfilingDirCleaner() as pdc:
                res = multiprocessing.Manager().dict()
                p = multiprocessing.Process(target=get_profiling, args=(body_desc, res))
                p.start()
                p.join(timeout=600)
                if p.is_alive():
                    logging.warning("Timeout for {}!".format(body_desc))
                    p.terminate()
                if not res.get("cmp", None) or not res.get("run_time", None):
                    raise RuntimeError("run error for {}!".format(body_desc))
                best_tuned_info["elapse"] = res["run_time"]
                if "args" not in best_tuned_info:
                    best_tuned_info["args"] = {}
        return best_tuned_info

    descs, _, _ = parallel_json_split(json.loads(desc_in))
    peels = [pt.composite_peel_analyze(desc) for desc in descs]
    logging.info("==============================Tune for plan==============================")
    candidates = []
    for peel in peels:
        peel_size, peel_dim_str, peel_spaces = _get_max_peel(peel)
        body_desc = peel.get_peeled_desc(peel_dim_str)  # only tune body.
        best_tuned_info = _tune(body_desc)
        candidates.append((peel_size, peel_spaces, best_tuned_info))
    block_fusion_pans = plan_block(candidates)
    logging.info("==============================Retune after plan==============================")
    final_tuned_info = []
    for (_, peel_str), peel in zip(block_fusion_pans, peels):
        body_desc = peel.get_peeled_desc(peel_str)
        best_tuned_info = _tune(body_desc)
        final_tuned_info.append(best_tuned_info)
    record_parallel_repo(desc_in, repo_path,
                         list(zip(final_tuned_info, block_fusion_pans)))
    logging.info("==============================Parallel tune done==============================")


def get_max_alloc(alloc_map):
    max = 0
    for i in alloc_map.values():
        if len(i) == 2 and max < i[1]:
            max = i[1]
    return max


def get_block_candidate_space(space):
    peel_spaces = []
    for peel_str in space:
        peel_list = [int(s) for s in peel_str.split(" ")]
        peel_space = [peel_list[i:i + 2] for i in range(0, len(peel_list), 2)]
        pre_axis = peel_space[0][0] - 1
        peel_size = 1
        for axis, peel in peel_space:
            if axis != pre_axis + 1:
                break
            pre_axis = axis
            peel_size *= peel
        if peel_size % 32 == 0:
            peel_spaces.append(peel_str)
    return peel_spaces


def remove_buffer_exceed_space(space, max_alloc):
    peel_spaces = []
    for peel_str in space:
        peel_list = [int(s) for s in peel_str.split(" ")]
        peel_space = [peel_list[i:i + 2] for i in range(0, len(peel_list), 2)]
        peel_size = 1
        for axis, peel in peel_space:
            peel_size *= peel
        # for ascend l1 buffer
        if max_alloc // peel_size <= 8388608:
            peel_spaces.append(peel_str)
    return peel_spaces


def tune_stitch_segment(desc_in: str, repo_path: str):
    descs, _, _, alloc_map, _, _ = stitch_json_split(json.loads(desc_in))
    max_alloc = get_max_alloc(alloc_map)
    peels = [pt.composite_peel_analyze(desc) for desc in descs]
    peeling_spaces = [p.get_peeling_space() for p in peels]
    best_tuning_info = {"run_time": float("inf"), "attrs": None}
    peeling_spaces = reduce(lambda x, y: set(x).intersection(set(y)), peeling_spaces)
    peeling_spaces = get_block_candidate_space(peeling_spaces)
    peeling_spaces = remove_buffer_exceed_space(peeling_spaces, max_alloc)
    logging.info("peeling_spaces: {}".format(peeling_spaces))
    for idx, peeling in enumerate(peeling_spaces):
        with ProfilingDirCleaner():
            try:
                peel_descs = [peels[i].get_peeled_desc(peeling) for i in range(len(descs))]
                # tuning for each split json
                tilings = []
                for desc_idx, peel_desc in enumerate(peel_descs):
                    logging.info("=============== Current peeling: {} [{}/{}], tuning sub json [{}/{}] =============="
                                 .format(peeling, idx + 1, len(peeling_spaces), desc_idx + 1, len(peel_descs)))
                    tilings.append(tuning_with_1block(peel_desc)["args"])

                attrs = {"peeling": peeling}
                for i, t in enumerate(tilings):
                    if not t:  # t is {}, which means the best tiling is auto tiling
                        continue
                    tiling_idx = "sub_attr_" + str(i + 1)
                    attrs[tiling_idx] = t

                # profiling in current config
                results = multiprocessing.Manager().dict()
                p = multiprocessing.Process(target=get_profiling, args=(desc_in, results, attrs))
                p.start()
                p.join(timeout=600)
                if p.is_alive():
                    logging.warning("Time out in current config: {}".format(attrs))
                    p.terminate()

                logging.info("=============== attrs: {}, profiling result: {} ===============".format(attrs, results))
                if results.get("cmp", None) is True and \
                        results.get("run_time", float("inf")) < best_tuning_info["run_time"]:
                    best_tuning_info["run_time"] = results["run_time"]
                    best_tuning_info["attrs"] = attrs
            except BaseException:
                logging.warning(traceback.format_exc())
    if best_tuning_info["run_time"] != float("inf"):
        logging.info("Tuning finished, best tuning info: {}".format(best_tuning_info))
        record_stitch_repo(desc_in, repo_path, best_tuning_info)
    else:
        logging.info("Tuning finished, but can not find a best tuning info for current json!")


def tune_composite_segment(json_str, repo_path=""):
    desc = json.loads(json_str)
    if "parallel_fusion" in desc:
        tune_parallel_segment(json_str, repo_path)
    elif "buffer_stitch" in desc:
        tune_stitch_segment(json_str, repo_path)


def tune_single_file(input_file, repo_path):
    if not input_file.endswith(".info") and not input_file.endswith(".json"):
        print("Skip {}, only process file with .info or .json suffix".format(input_file))
        return
    with open(input_file, 'r') as f:
        json_str = f.read()
        time_start = time.time()
        tune_composite_segment(json_str, repo_path)
        time_end = time.time()
        logging.debug("launch time: %f", time_end - time_start)


if __name__ == "__main__":
    try:
        if sys.argv[1] in ("-h", "--help"):
            sys.exit()
        input_str = sys.argv[1]
        if not os.path.isdir(input_str) and not os.path.isfile(input_str):
            logging.warning("ERROR: %s is not a directory or file.", input_str)
            sys.exit()
        options, args = getopt.getopt(sys.argv[2:], "-f-o:")
        skip_exist = True
        repo_path = None
        for option, value in options:
            if option == "-f":
                skip_exist = False
            elif option == "-o":
                repo_path = value
        if not repo_path:
            logging.warning("ERROR: Empty repository path.")
            sys.exit()
    except:
        print_usage()
        sys.exit()

    if os.path.isfile(input_str):
        tune_single_file(input_str, repo_path)
    else:
        files = os.listdir(input_str)
        for file in files:
            tune_single_file(input_str + "/" + file, repo_path)
