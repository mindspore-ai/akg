import os
import json
import argparse
import logging
import shutil
import glob

import akg
import auto_tune
from akg.utils.kernel_exec import ReturnType
from akg.composite.build_module import _update_attrs_gpu, _update_attrs_ascend
from akg.utils.op_test import random_data_to_disk

def set_environment(backend_str="", device_id=-1, device_total_num=-1):
    # how many multi-processing to build
    os.environ["BUILD_PARALLEL_NUM"] = "64"

    os.environ["PROFILING_MODE"] = "true"

    if backend_str == "aicore":
        # ascend config
        os.environ["RUNTIME_MODE"] = "air_cloud"
        if device_id != -1:
            os.environ["DEVICE_ID"] = str(device_id)
        if device_total_num != -1:
            os.environ["DEVICE_TOTAL_NUM"] = str(device_total_num)
        if os.environ.get("DEVICE_ID") is None:
            os.environ["DEVICE_ID"] = "4"
        if os.environ.get("DEVICE_TOTAL_NUM") is None:
            os.environ["DEVICE_TOTAL_NUM"] = "1"
        if os.environ.get("PROFILING_DIR") is None:
            curr_path = os.path.abspath("./")
            if not os.path.exists(curr_path + "/data"):
                os.makedirs(curr_path + "/data")
            os.environ["PROFILING_DIR"] = curr_path + "/data"
            print("Set profiling dir to {}".format(os.environ.get("PROFILING_DIR")))
    elif backend_str == "cuda":
        os.environ["RUNTIME_MODE"] = "gpu"
        # set the default gpu devices, plz never change it
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        # set the real devices you want to use
        os.environ["USE_GPU_DEVICES"] = "0,1,2,3"
    elif backend_str == "cpu":
        os.environ["RUNTIME_MODE"] = "cpu"
    else:
        raise ValueError("wrong backend_str={}".format(backend_str))

def get_all_infos_from_path(info_path):
    tmp_list = list()
    filelists = os.listdir(info_path)
    for fname in filelists:
        filepath = os.path.join(info_path, fname)
        if os.path.isdir(filepath):
            sub_list = get_all_infos_from_path(filepath)
            tmp_list.extend(sub_list)
        elif ".info" in fname:
            if "parallel" not in fname:
                tmp_list.append((os.path.abspath(info_path), fname))
    return tmp_list

def enable_input_cache():
    if os.environ.get("RANDOM_DATA_DISK_PATH", None) is None:
        os.environ["RANDOM_DATA_DISK_PATH"] = "."
    random_files = os.environ.get("RANDOM_DATA_DISK_PATH") + "/random_data*bin"
    if len(glob.glob(random_files)) == 0:
        random_data_to_disk(size=10485760, miu=[1, 0.5, 0.1], sigma=[0.1, 0.05, 0.01])

def parse_early_stop(task_options, early_stop):
    if early_stop == "auto":
        task_options.tuning_time_limit = 900
    elif early_stop == "never":
        task_options.tuning_time_limit = float("inf")
        task_options.best_time_limit = 0
        task_options.keep_updating_repo = True
    elif "speedup" in early_stop:
        try:
            speedup = float(early_stop.split("speedup")[0])
        except Exception as e:
            raise ValueError("Cannot parse {}: {}".format(early_stop, e))
        task_options.expect_speedup = speedup
    elif "cycles" in early_stop:
        try:
            best_time_limit = float(early_stop.split("cycles")[0])
        except Exception as e:
            raise ValueError("Cannot parse {}: {}".format(early_stop, e))
        task_options.best_time_limit = best_time_limit
    elif "s" in early_stop:
        try:
            tuning_time_limit = float(early_stop.split("s")[0])
        except Exception as e:
            raise ValueError("Cannot parse {}: {}".format(early_stop, e))
        task_options.tuning_time_limit = tuning_time_limit
    else:
        print("Invalid early_stop string: {}, please refer to help to set early_stop.".format(early_stop))
    return task_options

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--info_dir", type=str,
                        default="", help="info files dir")
    parser.add_argument("-f", "--file", type=str, default="",
                        help="info file name")
    parser.add_argument("-p", "--data_path", type=str,
                        default="None", help="path to use/save data")
    parser.add_argument("-pt", "--pt_model", type=str,
                        default=None, help="path to pretrained xgb model")
    parser.add_argument("-t", "--tuner_type", type=str, default="POLICY_ALL",
                        help="RADNOM/POLICY_ALL/ESA/GA/XGB_ALL")
    parser.add_argument("-l", "--tune_level", type=int, default=1,
                        help="tune level = 0, 1, (2)")           
    parser.add_argument("-n", "--new_space", type=int, default=1,
                        help="use new space = 0, 1")
    parser.add_argument("-r", "--repo_path", type=str, default=None,
                        help="the path to export repo")
    parser.add_argument("-g", "--gen_expect_mode", type=str, default="AKG",
                        help="the mode of generating expect: NUMPY or AKG")
    parser.add_argument("-tr", "--trails", type=int, default=None,
                        help="number of trails to search")
    parser.add_argument("-ct", "--cpu_threads", type=str, default="-1",
                        help="number of tune cpu threads, -1(means tune in [1,max]) or L,R(a range in [L,R])")
    parser.add_argument("-ic", "--input_cache", type=int, default=1,
                        help="whether to enable input cache during tuning, if enabled, random data will be generated under os.environ['RANDOM_DATA_DISK_PATH']")
    parser.add_argument("-ua", "--use_auto_tiling_init", type=int, default=0,
                        help="whether to use auto tiling config in init population, 0(no)/ 1(yes)")                        
    parser.add_argument("-m", "--mode", type=str, default="custom",
                        help="choose from [scan, online, offline, full], tuning_time: less->more; tuning result: slow->fast;")
    parser.add_argument("-es", "--early_stop", type=str, default="auto",
                        help="setting the condition to early stop, currently supports:\n" \
                             "1. Tuning Time Limit: Number+s, e.g. 60s, makes tuning stop after 1 minute\n" \
                             "2. Tuning Best Result Limit: 1) Number+cycles, e.g. 1000cycles, " \
                                 "makes tuning stop after finding a config that has less than 1000 cycles' runtime" \
                                 "Or 2) Number+speedupm, e.g. 1.5speedup, makes tuning stop after finding a config that" \
                                 "has 1.5 time speedup comparing to auto-tiling" \
                             "3. 'Never', makes tuning never early stop util all config in the space have been ran."
                        )
    parser.add_argument("-tops", "--test_get_op_scores", type=int, default=0,
                        help="whether to test op scores and generate stats, 0(no)/ 1(yes)")
    parser.add_argument("-et", "--enable_transfer", type=int, default=None)
    parser.add_argument("-ud", "--update_database", type=int, default=None,
                        help="Whether automatically updates database during tuning. If no value is set, only update during offline-tune.")


    args = parser.parse_args()
    
    all_files = list()
    if args.file != "":
        filename = args.file.split("/")[-1]
        path = os.path.abspath(args.file[:-len(filename)])
        all_files.append((path, filename))
    else:
        all_info_dir = args.info_dir.split(",")
        for d in all_info_dir:
            infos_list = get_all_infos_from_path(d)
            all_files.extend(infos_list)

    if args.tuner_type == "RANDOM": tuner_type = auto_tune.TunerType.RANDOM
    if args.tuner_type == "POLICY_ALL": tuner_type = auto_tune.TunerType.POLICY_ALL
    if args.tuner_type == "GA": tuner_type = auto_tune.TunerType.GA
    if args.tuner_type == "ESA": tuner_type = auto_tune.TunerType.ESA
    if args.tuner_type == "XGB_ALL": tuner_type = auto_tune.TunerType.XGB_ALL_SPACE

    data_path = None if args.data_path == "None" else args.data_path

    gen_expect_mode = auto_tune.GenExpectMode.NUMPY if args.gen_expect_mode == "NUMPY" else auto_tune.GenExpectMode.AKG
    task_schedule_mode = "auto"
    if args.update_database is None:
        args.update_database = args.mode in ["offline", "full"]
    # task options and common settings
    logger = logging.getLogger("test_akg_tuning")
    if args.new_space:
        from akg.composite import generate_trait
        task_options = auto_tune.TaskOptions(tune_level=args.tune_level,
                                             skip_exist=len(all_files) > 1,
                                             generate_trait=generate_trait,
                                             auto_rm_log=False,
                                             tuner_type=tuner_type,
                                             keep_updating_repo=len(all_files) == 1,
                                             cpu_threads=args.cpu_threads,
                                             use_auto_tiling_init=args.use_auto_tiling_init,
                                             update_database=args.update_database)
        if args.mode == "custom":
            task_schedule_mode = "custom"
            if args.tuner_type == "POLICY_ALL":
                task_options.trials = 320
            if args.trails is not None:
                task_options.trials = args.trails
            if args.repo_path is None:
                task_options.repo_path = "tuner_" + args.tuner_type + "_trial" + str(task_options.trials) + "_lv" + str(args.tune_level) + "_repo.json"
            else:
                task_options.repo_path = args.repo_path
            
            if args.pt_model is not None:
                task_options.pt_model_path = args.pt_model

            if args.early_stop == "never" and args.tuner_type != "POLICY_ALL":
                raise ValueError("Cannot 'never' early stop for tuner {}, please reset early_stop or reset tuner_type.")
            task_options = parse_early_stop(task_options, args.early_stop)
        else:
            task_options.config_to_mode(args.mode)
            if args.early_stop != "auto":
                task_options = parse_early_stop(task_options, args.early_stop)
            if args.repo_path is not None:
                task_options.repo_path = args.repo_path
        logger = auto_tune.get_logger(task_options.tuning_log_path).getChild("test_akg_tuning")
    else:
        task_options = None
    
    if args.input_cache:
        enable_input_cache()

    if args.enable_transfer is None:
        args.enable_transfer = task_options.enable_transfer
    else:
        task_options.enable_transfer = args.enable_transfer

    progress_record = auto_tune.ProcessRecord(len(all_files), "ALL-INFO-TUNING")
    stat = {"Score": [], "Op": [], "Speedup": []}
    transfer_info = {}
    if os.path.exists("transfer_info.txt"):
        with open("transfer_info.txt", "r") as f:
            transfer_info = json.loads(f.read())
        print("load transfer info {}".format(transfer_info))
    for i, (path, info_file) in enumerate(all_files):
        logger.info("Begin tune No.{}/{} files: [{}]".format(i+1, len(all_files), info_file))
        with open(path + "/" + info_file, 'r') as f:
            desc = f.read()
        
        desc_d = json.loads(desc)
        backend = desc_d["process"]
        set_environment(backend)
        
        attrs = {"use_new_space": bool(args.new_space)}
        if task_options is not None:
            task_options.attrs = attrs
            if args.early_stop == "auto":
                task_options.set_best_time_limit()
            task_options.wait_when_cpu_busy = True
        if backend == "cuda":
            attrs = _update_attrs_gpu(desc_d, attrs, True)
        elif backend == "cpu":
            if args.early_stop == "auto":
                task_options.set_best_time_limit(0)
        else:
            attrs = _update_attrs_ascend(desc_d, attrs)

        logger.info("Attrs: {}".format(attrs))
        if args.test_get_op_scores:
            from .test_tuning_composites import test_get_op_scores
            ret = test_get_op_scores(desc,
                                     task_options,
                                     task_options.tuner_type,
                                     data_path,
                                     gen_expect_mode)
            for k in op_score_stats:
                if k not in ret:
                    break
                op_score_stats[k].append(ret[k])
        elif args.new_space:
            logger.info("len of transfer info = {}".format(len(transfer_info)))
            tuner = auto_tune.tune_composite_v2(desc,
                                                task_options,
                                                task_options.tuner_type,
                                                data_path,
                                                gen_expect_mode,
                                                return_tuner=True,
                                                task_schedule_mode=task_schedule_mode,
                                                transfer_info=transfer_info)
            if isinstance(tuner, dict):
                best_config = tuner
            else:
                best_config = tuner._best_config
                if args.enable_transfer:
                    transfer_info.update(tuner.transfer_info)
                    logger.info("update {} transfer info, curr size = {}".format(len(tuner.transfer_info), len(transfer_info)))

        else:
            raise ValueError("Previous version tuning is not supported, please set -n=1")
        logger.info("FILE {}, BEST CONFIG = {}".format(info_file, best_config))


        progress_record.update()

    if os.path.exists(task_options.tuning_log_path) and task_options.auto_rm_log:
        print("Finish tuning, exit and delete log files {}".format(task_options.tuning_log_path))
        shutil.rmtree(task_options.tuning_log_path)

    if args.test_get_op_scores:
        all_scores = list(op_score_stats["Score"])
        sorted_score_idx = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
        print("=========== Score Rank ==================")
        for i, idx in enumerate(sorted_score_idx):
            print("Rank {}: Score = {}, Speedup = {}, Op = {}".format(i, op_score_stats["Score"][idx], op_score_stats["Speedup"][idx], op_score_stats["Op"][idx]))

        all_speedup = list(op_score_stats["Speedup"])
        sorted_speedup_idx = sorted(range(len(all_speedup)), key=lambda k: all_speedup[k], reverse=True)
        print("=========== Speedup Rank ==================")
        for i, idx in enumerate(sorted_speedup_idx):
            print("Rank {}: Score = {}, Speedup = {}, Op = {}".format(i, op_score_stats["Score"][idx], op_score_stats["Speedup"][idx], op_score_stats["Op"][idx]))


    if len(transfer_info) > 0:
        # dump transfer info
        light_info = {}
        for json_str, value in transfer_info.items():
            try:
                light_info[json.loads(json_str).get("op", "unknown")] = value
            except:
                light_info[json_str] = value
        with open("transfer_info.txt", "w") as f:
            s = json.dumps(light_info, sort_keys=True, indent=4)
            f.write(s)

