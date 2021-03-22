from collections import namedtuple
import os
import logging


def get_block_str_from_config(config: namedtuple):
    block_param = ""
    if "block_x" in getattr(config, "_fields"):
        block_param += str(config.block_x) + " "

    if "block_y" in getattr(config, "_fields"):
        block_param += str(config.block_y) + " "

    if "block_z" in getattr(config, "_fields"):
        block_param += str(config.block_z) + " "
    return block_param


def get_thread_str_from_config(config: namedtuple):
    thread_param = ""
    if "thread_x" in getattr(config, "_fields"):
        thread_param += str(config.thread_x) + " "

    if "thread_y" in getattr(config, "_fields"):
        thread_param += str(config.thread_y) + " "

    if "thread_z" in getattr(config, "_fields"):
        thread_param += str(config.thread_z) + " "
    return thread_param


def get_parallel_build_num():
    """get the num of parallel build"""
    env_dic = os.environ
    try:
        return int(env_dic.get('BUILD_PARALLEL_NUM').lower()) if env_dic.get('BUILD_PARALLEL_NUM') else 1
    except NameError as e:
        logging.error(e)
        return 1


def get_available_gpu_num():
    """get the num of gpu"""
    env_dic = os.environ
    try:
        return [int(id) for id in env_dic.get('USE_GPU_DEVICES').split(",")] if env_dic.get('USE_GPU_DEVICES') else [0, ]
    except NameError as e:
        logging.error(e)
        return 1

def get_real_attr(value ,key ,need_tune_json, need_tune_keys):
    if key not in need_tune_keys:
        return value
    if need_tune_json[key]['dtype'] == "bool":
        if need_tune_json[key]['options'][value].lower()  == "true":
            return True
        elif need_tune_json[key]['options'][value].lower()  == "false":
            return False
        else:
            raise TypeError("Wrong boolean type, please check json file")
    elif need_tune_json[key]['dtype'] == "str":
        if isinstance(need_tune_json[key]['options'][value], str):
            return need_tune_json[key]['options'][value]
        else:
            raise TypeError("Wrong str type, please check json file")
    elif need_tune_json[key]['dtype'] == "int":
        if isinstance(need_tune_json[key]['options'][value], int):
            return need_tune_json[key]['options'][value]
        else:
            raise TypeError("Wrong int type, please check json file")


def merge_attrs(attrs, config, need_tune_json):
    tiling = [getattr(config, name) for name in getattr(
            config, '_fields') if name.startswith('tiling')]
    dim_str = ''
    d_config = config._asdict()
    d_attrs = attrs._asdict()
    
    is_2d_tiling = False
    for name in getattr(config, '_fields'):
        if name.startswith('tiling'):
            if name.count("_") == 2:
                is_2d_tiling = True
            break
    
    for i, element in enumerate(tiling):
        if is_2d_tiling:
            if i % 2 == 0:
                dim_str += "0 " + str(i//2) + " "
            dim_str += str(element) + " "
        else:
            # 1d tiling
            dim_str += "0 " + str(i) + " " + str(element) + " 1 "

    # add block, thread information
    block = [str(getattr(config, name)) for name in getattr(
        config, '_fields') if name.startswith('block')]
    bind_block_str = ' '.join(block)

    thread = [str(getattr(config, name)) for name in getattr(
        config, '_fields') if name.startswith('thread')]
    bind_thread_str = ' '.join(thread)

    d_attrs['dim'] = dim_str
    d_attrs['bind_block'] = bind_block_str
    d_attrs['bind_thread'] = bind_thread_str

    need_tune_keys = need_tune_json.keys()
    for key in need_tune_keys:
        d_attrs[key] = d_config[key]

    # make a new attrs with config info
    attrs_type = type(attrs)
    config_list = [get_real_attr(d_attrs[k],k,need_tune_json, need_tune_keys) for k in d_attrs]
    new_attrs = attrs_type(*config_list)
    return new_attrs


def get_skip_configs_from_log(skip_configs_log):
    skip_config_set = set()
    if skip_configs_log != "":
        with open(skip_configs_log, 'r') as file:
            for line in file:
                config = str(line.split("|")[1]).strip()
                skip_config_set.add(config)
            print("SKIP CONFIGS NUMBER:", len(skip_config_set))
    return skip_config_set

def get_tuning_attrs_from_json(tuning_attrs_json):
    import json
    need_tune_spaces = [[]]
    keys = []
    json_string = dict()
    if tuning_attrs_json != "":
        with open(tuning_attrs_json,'r') as file:
            json_string =json.load(file)
            for key in json_string.keys():
                keys.append(key)
                num_options = len(json_string[key]['options'])
                tmp_spaces = []
                for space in need_tune_spaces:
                    for i in range(num_options):
                        tmp_space = space[:]
                        tmp_space.append(i)
                        tmp_spaces.append(tmp_space)
                need_tune_spaces = tmp_spaces[:]
    return (keys, need_tune_spaces, json_string)

if __name__ == "__main__":
    """test components"""
    file_name = "tuning_attrs_descs/reduce_tuning_attrs_desc.json"
    keys, need_tune_spaces = get_tuning_attrs_from_json(file_name)
    print(keys)
    print(need_tune_spaces)