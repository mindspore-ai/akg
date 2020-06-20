#!/bin/bash

# Copyright 2019 Huawei Technologies Co., Ltd
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

cd $(dirname $0)

. ../test_env.sh kc_air

export RANDOM_DATA_DISK_PATH=$(pwd)
TVM_CASE_ROOT_HOME="${RANDOM_DATA_DISK_PATH}/.."
export WRITE_TO_DISK=No
python -c  "import tensorio;tensorio.random_data_to_disk(size=10485760, miu=[1, 0.5, 0.1], sigma=[0.1, 0.05, 0.01])"
log_ftp_path="./"
pipe_line_type=""
pipe_line_index=0
max_pipe_line=1
all_case_run_flag="true"
rpc_host=""
rpc_port=0
runtime_mode=""
sub_pipe_flag=""

while getopts ":d:l:i:m:a:h:p:r:s:" opt
do
    case $opt in
        d) # 日志路径
        log_ftp_path=$OPTARG
        ;;
        l) # 流水线类型
        pipe_line_type=$OPTARG
        ;;
        i) # 流水线索引，拆分时使用
        pipe_line_index=$OPTARG
        ;;
        m) # 流水线最大数量，拆分时使用
        max_pipe_line=$OPTARG
        ;;
        a) # 是否跑完全流水线标记
        all_case_run_flag=$OPTARG
        ;;
        h)  # rpc_host
        rpc_host=$OPTARG
        ;;
        p) # rpc_port
        rpc_port=$OPTARG
        ;;
        r) # runtime_mode
        runtime_mode=$OPTARG
        ;;
        s) # sub_pipe_flag
        sub_pipe_flag=$OPTARG
        ;;
        ?)
        echo "valid optarg"
        exit 1;;
    esac
done

echo "log_ftp_path:${log_ftp_path}, pipe_line_type:${pipe_line_type}, pipe_line_index:${pipe_line_index}, max_pipe_line:${max_pipe_line}, sub_pipe_flag:${sub_pipe_flag}, all_case_run_flag:${all_case_run_flag}"

echo "RUNTIME_MODE:${RUNTIME_MODE}"

taskid=$(date +%s%N | md5sum | head -c 32)
log_file_path="${log_ftp_path}/${taskid}/"
mkdir -p "${log_file_path}"
echo "log_file_path:${log_file_path}"

all_case_list_file=${log_file_path}/all_case_list_${pipe_line_type}_${pipe_line_index}_${taskid}.txt
touch "${all_case_list_file}"
case_result_file=${log_file_path}/case_result_${pipe_line_type}_${pipe_line_index}_${taskid}.csv
touch "${case_result_file}"
all_case_count_file=${log_file_path}/all_case_count_${pipe_line_type}_${pipe_line_index}_${taskid}.csv
touch "${all_case_count_file}"

exit_flag=0

function exit_with_code()
{
    local function_name=$1
    local ret=$2
    local case_name=$3
    local output_file=$4
    if [[ "${ret}" != "0" && ${all_case_run_flag} == "false" ]] ; then
        echo "${function_name}, ${case_name}, ${ret}, failed , exit"
        exit_flag=1
        [ -f "${output_file}" ] && cat "${output_file}"
        exit ${exit_flag}
    fi
}

function get_case_list_by_file()
{

    local path=$1
    local attr=$2
    old_path=$(pwd)
    cd "${path}" || return
    chmod 400 ./*.py
    rm -rf "${all_case_list_file}"
    touch "${all_case_list_file}"
    local line_idx=0
    local attr_flag=$(echo "${attr}" | awk '{print $2}')
    for file_name in test_*.py
    do
        attr_flag_count=$(grep -c ".mark.${attr_flag}" "${file_name}")
        if [[ ${attr_flag_count} -gt 0 ]]
        then
            line_res=$(( line_idx % max_pipe_line ))
            ((line_idx++))
            if [[ ${line_res} -eq ${pipe_line_index} ]]
            then
                echo "${file_name%%.py*}" >> "${all_case_list_file}"
            fi
        fi
    done
    cd "${old_path}" || return
    echo
    cat "${all_case_list_file}"
}

function get_case_list_by_args()
{
    local path=$1
    local case_name=$2
    local attr=$3
    old_path=$(pwd)
    cd "${path}" || return
    chmod 400 "${case_name}.py"
    rm -rf "${all_case_list_file}"
    touch "${all_case_list_file}"
    export CASE_ARG_ATTR="${attr}"
    echo "attr:${attr}, CASE_ARG_ATTR:${CASE_ARG_ATTR}, case_name:${case_name}"
    python -c "import ${case_name};${case_name}.print_args()" > "./tmp_case_list.txt" 2>&1
    ret=$?
    unset CASE_ARG_ATTR
    exit_with_code "get_case_list_by_args" "${ret}" "NULL"  "./tmp_case_list.txt"

    local line_idx=0
    cat ./tmp_case_list.txt
    cat ./tmp_case_list.txt |grep "${case_name}&" > ./tmp_case_list_${pipe_line_type}_${pipe_line_index}_${taskid}.txt

    while read -r line
    do
        line_res=$(( line_idx % max_pipe_line ))
        ((line_idx++))
        if [[ ${line_res} -eq ${pipe_line_index} ]] ;then
            echo "${line}" >> "${all_case_list_file}"
        fi
    done < "./tmp_case_list_${pipe_line_type}_${pipe_line_index}_${taskid}.txt"
    cd "${old_path}" || return
}

function run_one_case()
{
    local path=$1
    local attr=$2
    local case_name=$3
    local case_flag=$4
    local test_index=$5

    old_path=$(pwd)
    cd "${path}" || return
    echo "$(date '+%Y-%m-%d %H:%M:%S') attr:${attr}, case_name:${case_name}, case_flag:${case_flag}, test_index:${test_index}"
    chmod 400 "${case_name}.py"
    start_case_time=$(date +%s)
    pytest  "${attr}" -s "${case_name}.py"  > "./${case_name}-${case_flag}-${taskid}.log" 2>&1
    ret=$?
    end_case_time=$(date +%s)
    time_ring=$(( end_case_time - start_case_time ))
    result="succ"
    if [[ ${ret} -ne 0 ]] ; then
        result="fail"
    fi

    grep "func_time_required" "./${case_name}-${case_flag}-${taskid}.log"
    echo "${case_name}-${case_flag}, ${time_ring}"

    echo "${pipe_line_type}, ${sub_pipe_flag}, ${case_name}-${case_flag},${result},${case_name}-${case_flag}-${taskid}.log,${ret},${time_ring}" >> "${case_result_file}"
    if [[ "x${result}" != "xsucc" ]] ;then
        cd "${old_path}" || return
        echo "${case_name}, ${case_flag}, ${ret}, failed "
	echo "################# ${case_name}-${case_flag}-${taskid}.log start ##################### "
        cat  "${case_name}-${case_flag}-${taskid}.log"
	echo "################# ${case_name}-${case_flag}-${taskid}.log end   ##################### "
	if [[ "x${result}" == "xfail" ]] ;then
            # 可预期错误
            exit_with_code "run_one_case" ${ret} "${case_name}-${case_flag}" "NULL"
        fi
    fi

    cd "${old_path}" || return
    return 0
}

function run_operator_case_list()
{
    local path=$1
    local attr=$2
    get_case_list_by_file "${path}" "${attr}"
    case_flag="daily"
    old_path=$(pwd)
    cd "${path}" || return
    while read -r case_name
    do
        run_one_case "${path}" "${attr}" "${case_name}" "${case_flag}" "0"
    done < "${all_case_list_file}"
    cd "${old_path}" || return
}

function run_net_case_list()
{
    local path=$1
    local attr=$2
    local case_name=$3

    if [[ "${RUNTIME_MODE}" == "rpc" ]]
    then
        attr="-m rpc"
    elif [[ "${RUNTIME_MODE}" == "rpc_cloud" ]]
    then
        attr="-m rpc_cloud"
    else
        attr="-m level0"
    fi

    old_path=$(pwd)
    run_attr=$(echo "${attr}" | awk -F "=" '{print $2}')
    get_case_list_by_args "${path}" "${case_name}" "${run_attr}"
    cd "${path}" || return
    while read -r line
    do
    	echo "${line}"
        case_name=$(echo "${line}" | awk -F "&" '{print $1}')
        test_index=$(echo "${line}" | awk -F "&" '{print $2}')
        case_flag=$(echo "${line}" | awk -F "&" '{print $3}')
        export TEST_INDEX=${test_index}
        run_one_case "${path}" "${attr}" "${case_name}" "${case_flag}" "${test_index}"
        unset TEST_INDEX
    done < "${all_case_list_file}"
    cd "${old_path}" || return
}

function run_unittest_list()
{
    local path=$1
    local attr=$2
    local case_pipe=$3
    old_path=$(pwd)
    cd "${path}" || return
    local case_file_list=("cce/test_bias_add.py" "pass/test_promote_if.py" "pass/test_sink_if.py" "pass/test_ir_parser.py" "pass/test_elim_vector_mask.py" "pass/test_copy_propagation.py")
    for case_name in "${case_file_list[@]}"
    do
        python "${case_name}" > "${case_name}.log" 2>&1
        ret=$?
        if [[ "${ret}" -ne "0" ]] ; then
            echo "run unit case ${case_name} failed!!!"
            cp -ar "${case_name}.log" "${log_file_path}"
            result="fail"
        else
            result="succ"
        fi
        echo "${pipe_line_type}, ${sub_pipe_flag}, ${case_name},${result},${case_name}.log,${ret},0,null" >> "${case_result_file}"
        exit_with_code "run_unittest_list" "${ret}" "${case_name}.log" "NULL"
    done
    cd "${old_path}" || return
}

if [[ "${RUNTIME_MODE}" == "rpc" ]]
then
    run_attr="rpc"
elif [[ "${RUNTIME_MODE}" == "rpc_cloud" ]]
then
    run_attr="rpc_cloud"
else
    run_attr="level0"
fi

if [[ "${pipe_line_type}" == "vector_level0" ]]
then
     run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/vector" "-m level0"
elif [[ "${pipe_line_type}" == "vector_level1" ]]
then
     run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/vector" "-m level1"
elif [[ "${pipe_line_type}" == "vector_aic_model" ]]
then
    run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/vector" "-m aicmodel"
elif [[ "${pipe_line_type}" == "vector_rpc_mini" ]]
then
     run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/vector" "-m rpc_mini"
elif [[ "${pipe_line_type}" == "vector_rpc_cloud" ]]
then
     run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/vector" "-m rpc_cloud"
elif [[ "${pipe_line_type}" == "cube_level0" ]]
then
     run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/cube" "-m level0"
elif [[ "${pipe_line_type}" == "cube_level1" ]]
then
     run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/cube" "-m level1"
elif [[ "${pipe_line_type}" == "cube_rpc_mini" ]]
then
     run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/cube" "-m rpc_mini"
elif [[ "${pipe_line_type}" == "cube_rpc_cloud" ]]
then
     run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/cube" "-m rpc_cloud"
elif [[ "${pipe_line_type}" == "bert_pipe" ]]
then
    run_net_case_list "${TVM_CASE_ROOT_HOME}/perf_benchmark" "-m level0" "test_bert_all_001"
elif [[ "${pipe_line_type}" == "ssd_pipe" ]]
then
    run_net_case_list "${TVM_CASE_ROOT_HOME}/perf_benchmark" "-m level0" "test_ssd_all_001"
elif [[ "${pipe_line_type}" == "reid_pipe" ]]
then
    run_net_case_list "${TVM_CASE_ROOT_HOME}/perf_benchmark" "-m level0" "test_reid_all_001"
elif [[ "${pipe_line_type}" == "resnet50_pipe" ]]
then
    run_net_case_list "${TVM_CASE_ROOT_HOME}/perf_benchmark" "-m level0" "test_resnet50_all_001"
elif [[ "${pipe_line_type}" == "alexnet_pipe" ]]
then
    run_net_case_list "${TVM_CASE_ROOT_HOME}/perf_benchmark" "-m level0" "test_alexnet_all_001"
elif [[ "${pipe_line_type}" == "lenet_pipe" ]]
then
    run_net_case_list "${TVM_CASE_ROOT_HOME}/perf_benchmark" "-m level0" "test_lenet_all_001"
elif [[ "${pipe_line_type}" == "autodiff_pipe" ]]
then
    run_net_case_list "${TVM_CASE_ROOT_HOME}/perf_benchmark" "-m level0" "test_autodiff_all_001"
elif [[ "${pipe_line_type}" == "allnet_pipe" ]]
then
    run_net_case_list "${TVM_CASE_ROOT_HOME}/perf_benchmark" "-m level0" "test_allnet_all_001"
elif [[ "${pipe_line_type}" == "unittest_pipe" ]]
then
    run_unittest_list "${TVM_CASE_ROOT_HOME}/unittest" "-m level0" "unittest"
elif [[ "${pipe_line_type}" == "dynamic_shape" ]]
then
    run_operator_case_list "${TVM_CASE_ROOT_HOME}/operators/dynamic_shape" "-m level0"
else :
    echo "not support ${pipe_line_type}"
    exit_flag=1
    exit ${exit_flag}
fi

succ_case_count=$(< "${case_result_file}" awk -F "," '{print $4}' | grep -c "succ" )
fail_case_count=$(< "${case_result_file}" awk -F "," '{print $4}' | grep -c "fail" )
all_case_count=$(< "${case_result_file}" wc -l)

echo "${pipe_line_type},${sub_pipe_flag},${all_case_count},${succ_case_count},${fail_case_count}" > "${all_case_count_file}"

cat "${case_result_file}"
echo "pipe_line_type:${pipe_line_type},sub_pipe_flag:${sub_pipe_flag},all_case_count:${all_case_count},succ_case_count:${succ_case_count},fail_case_count:${fail_case_count}, log_file_path:${log_file_path}"
exit ${exit_flag}
