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

"""Test Base class"""
import os
import sys
import time
import tarfile
import datetime
import collections
import numpy as np
from akg import dim
from akg.utils.result_analysis import count_unequal_element
from tests.common import tensorio
from tests.common.ftp_handel import ftpHandle
from tests.common.log import Log

PERFORMANCE_TEST = "PERFORMANCE_TEST"


class TestBase(object):
    pandora_logger_ = None

    def params_init(self, case_name, case_path, max_retry=3):
        self.casename = case_name
        self.caselog_path = case_path
        self.max_retry = max_retry
        # Define the log storage location, which is stored in case_log by default.
        self.case_result = True
        if TestBase.pandora_logger_ is None:
            TestBase.pandora_logger_ = Log(case_name, case_path)
        self._log = TestBase.pandora_logger_.log
        self.test_args = []
        self.caseresult = True
        self._exception = None

    def setup(self):
        self._log.info("TestBase:{0} Setup case".format(self.casename))
        return True

    def teardown(self):
        self._log.info("TestBase:{0} Teardown".format(self.casename))
        return

    def run_test_arg_func(self, test_args=[], attr=None):
        if not attr:
            self._log.info("attr is None")
            return False
        run_mode = self.get_env_var("RUNTIME_MODE")
        if run_mode in ["compile_cloud", "compile_mini"]:
            mode = "compile"
        else:
            mode = "execute"
        for arg in test_args:
            self._log.info(arg)
            if attr in arg[-1]:
                case_result, exception = self.common_run([arg[0:-1]], mode=mode)
                if not case_result:
                    self._log.info("{0} run failed".format(arg))
                    return False
        return True

    def print_args(self):
        for index, arg in enumerate(self.test_args):
            print("{0} {1}".format(index, arg[0]))

    def ana_args(self, arg, is_conv=False):
        caseflag, func, args = arg[0:3]
        kwargs = {}
        attrs = self.get_dim_info(arg, is_conv)
        if self.get_env_var(PERFORMANCE_TEST):
            attrs["record_core"] = True
        if attrs is not None:
            if len(arg) == 5 and not arg[-1]:
                args = list(args)
                args.append(attrs)
                args.append(arg[-1])
                kwargs = {}
            else:
                args = list(args)
                kwargs = {"attrs": attrs}
        return caseflag, func, args, kwargs

    def get_dim_info(self, arg, is_conv=False):
        info = dim.Dim()
        tile_dims = []
        dims = None
        enable_multicore = None
        dynamic = False
        partial_dynamic = False
        bypass_l1 = False
        if "dynamic" in arg:
            dynamic = True
            if isinstance(arg, tuple):
                arg = list(arg)
                arg.remove("dynamic")
                arg = tuple(arg)
            else:
                arg.remove("dynamic")
        if "partial_dynamic" in arg:
            partial_dynamic = True
            arg.remove("partial_dynamic")
        if "bypassL1" in arg:
            bypass_l1 = True
            arg.remove("bypassL1")
        if is_conv:
            dy = dynamic or partial_dynamic
            if len(arg) == 4:
                conv_tile = arg[3]
                if len(conv_tile) > 0:
                    if not dy:
                        return {
                            "dim": str(info),
                            "conv_tile": conv_tile,
                            "enable_multicore": True,
                            "bypass": 1 if bypass_l1 else 0,
                        }
                    else:
                        return {
                            "dim": str(info),
                            "conv_tile": conv_tile,
                            "dynamic": dynamic,
                            "partial_dynamic": partial_dynamic,
                            "bypass": 1 if bypass_l1 else 0,
                        }
            elif dy and len(arg) == 3:
                return {
                    "dynamic": dynamic,
                    "partial_dynamic": partial_dynamic,
                    "bypass": 1 if bypass_l1 else 0,
                }

        if len(arg) == 5 and not arg[-1]:
            dims = arg[3]
            for d in range(len(dims)):
                tile_dims.append(dims[d][0])
        elif (len(arg) == 5 and arg[-1]) or len(arg) == 4:
            if isinstance(arg[3], (bool, int)):  # only multicore info
                enable_multicore = arg[3]
            elif isinstance(arg[3][-1], (bool, int)):  # dim info and multicore info
                enable_multicore = arg[3][-1]
                dims = arg[3][0]
            else:  # only dim info
                dims = arg[3]
            if dims is not None:
                for i in range(len(dims)):
                    if (isinstance(dims[i][0], int)):
                        # only one index, ((l1,l0),(l1,l0),...)
                        i_dims = dims
                    else:
                        # multiple indices, (((l1,l0),(l1,l0),...), ((l1,l0),(l1,l0),...))
                        i_dims = dims[i]

                    for d in range(len(i_dims)):
                        info.setdim(index=i,
                                    axis=d,
                                    tilel1=i_dims[d][0],
                                    tilel0=i_dims[d][1])

        if len(arg) == 5 and not arg[-1]:
            return {"tile": tile_dims}
        else:
            res = {"dim": str(info), "dynamic": dynamic}
            if enable_multicore:
                res["enable_multicore"] = enable_multicore
            return res

    def get_env_var(self, env_key=None):
        env_dic = os.environ
        env_var = env_dic.get(env_key)
        if env_var:
            return env_var
        return None

    def translate_func_name(self, arg):
        args_list = []
        args_list.append(arg[0])
        func = arg[1]
        if isinstance(func, str):
            args_list.append(func)
        else:
            args_list.append(func.__name__)
        for i in range(2, len(arg)):
            args_list.append(arg[i])
        return tuple(args_list)

    def import_get_func(self, func, mode):
        """
        from test_run.tile_run import tile_compile
        :param func: function name
        :param mode: case mode
        :return:
        """
        func_fromlist = "tests.common.test_run." + func
        try:
            new_func = func
            func_py = __import__(func_fromlist, fromlist=func)
            run_func = getattr(func_py, new_func)
        except (ImportError, AttributeError) as e:
            new_func = func.split("_run")[0] + "_" + mode
            func_py = __import__(func_fromlist, fromlist=new_func)
            run_func = getattr(func_py, new_func)
        return run_func

    def common_run(self, args, dtype_list=None, mode="execute", is_conv=False, raise_exception=True):
        """
        :param dtype_list:operator program data type
        :param mode: operator run mode: such as rpc_cloud/aicmodel
        :param raise_exception: By default, when an exception occurs in the compilation,
                                the assert is used to interrupt the program.
        :return:
        """
        for arg in args:
            starttime = datetime.datetime.now()
            caseflag, func, args, kwargs = self.ana_args(arg, is_conv)

            if dtype_list:
                if not self.set_args_dtype(args, func, dtype_list):
                    self._log.error("common_run failed for set_args_dtype")
                    return False

            if isinstance(func, str):
                self._log.info("common_run :: run {funcname} with args:{args}".format(funcname=func, args=args))
                func = self.import_get_func(func, mode)
            else:
                self._log.info("common_run :: run {funcname} with args:{args}".format(funcname=func.__name__, args=args))

            mod = None
            if mode == "compile":
                try:
                    mod = func(*args, **kwargs)
                except Exception as e:
                    TestBase.pandora_logger_.traceback()
                    self._exception = e
                finally:
                    if (not mod) or self._exception:
                        self._log.error("common_run :: circle {0} fail !".format(self.translate_func_name(arg)))
                        self._log.error("common_run :: compile failed !")
                        self.case_result = False

            elif mode == "execute":
                input, output, expect, runres = func(*args, **kwargs)
                rtol = atol = 0
                compare_res = []
                if isinstance(runres, list):
                    if isinstance(runres[-1], (list, tuple)):
                        rtol = runres[-1][0]
                        atol = runres[-1][1]
                        runres = list(runres[:-1])
                    compare_res = runres
                    runres = all(runres)
                elif isinstance(runres, collections.Iterable):
                    compare_res = list(runres)
                else:
                    compare_res = [runres]

                kernel_name = self.get_kernel_name(args, func)
                cce_file_name = self.collect_cce(kernel_name)
                ir_file_name = self.collect_ir(kernel_name)
                if not runres:
                    runtime_mode = os.environ.get("RUNTIME_MODE")
                    if runtime_mode in ["rpc", "rpc_cloud", "air", "air_cloud"]:
                        for retry in range(self.max_retry):
                            self._log.error("Case result is incorrect, but RPC server occasionally produce incorrect "
                                            "output. Retry it before reporting failure. Retry count: " + str(retry + 1))
                            input, output, expect, runres = func(*args, **kwargs)

                            if isinstance(runres, list):
                                if isinstance(runres[-1], (list, tuple)):
                                    rtol = runres[-1][0]
                                    atol = runres[-1][1]
                                    runres = list(runres[:-1])
                                compare_res = runres
                                runres = all(runres)
                            elif isinstance(runres, collections.Iterable):
                                compare_res = list(runres)
                            else:
                                compare_res = [runres]

                            if runres:
                                break
                if not runres:
                    self._log.error("common_run :: circle {0} fail !".format(self.translate_func_name(arg)))
                    self._log.error("common_run :: CompareResult: %s", str(compare_res))

                    if rtol == 0:
                        self._log.error("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                        self._log.error("Caution: the 'rtol' and 'atol' is default $$$$$1e-4$$$$$")
                        self._log.error("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                        rtol = atol = 1e-4

                    if isinstance(expect, (tuple, list)):
                        for i, tmp in enumerate(expect):
                            count_unequal_element(tmp, output[i], rtol, atol)
                    else:
                        if not isinstance(expect, np.ndarray):
                            expect = np.atleast_1d(expect)
                        count_unequal_element(expect, output, rtol, atol)

                    if not self.collect_data(input, output, cce_file_name, ir_file_name, arg, kernel_name):
                        self._log.error("common_run :: collect data failed")
                    self.case_result = False
                else:
                    self._log.info("common_run :: circle {0} pass !".format(self.translate_func_name(arg)))
                    if cce_file_name and os.path.exists(cce_file_name):
                        os.remove(cce_file_name)
                    if ir_file_name and os.path.exists(ir_file_name):
                        os.remove(ir_file_name)
                    self.case_result &= True

            endtime = datetime.datetime.now()
            self._log.info("{0} testcase use ***Running Time*** is: {1}s. "
                           .format(caseflag, (endtime - starttime).seconds))
        self._log.info(self.case_result)
        '''
        use assert in the common_run function:
        Because the common_run function in the use cases does not verify the return value, the result cannot be 
        printed normally after the program ends, so the execution result needs to be judged in the common_run function.
        '''
        if (not self.case_result) and raise_exception:
            assert self.case_result
        return self.case_result, self._exception

    def get_args_dtype(self, input_args_names):
        """
        Get the dtype of the function input parameter, return its index
        :param input_args_names: Test operator method
        :return: kernel_name
        """
        return tuple([index for index, name in enumerate(input_args_names) if str(name).__contains__("dtype")])

    def get_kernel_name(self, args, func):
        func_input_args_names = func.__code__.co_varnames
        kernel_name = func.__name__.split('_run')[0].split('_execute')[0]
        for index, name in enumerate(func_input_args_names):
            if str(name).__contains__("kernel_name"):
                kernel_name = func.__name__.split('_run')[0].split('_execute')[0]
                break
        return kernel_name

    def replace_args_dtype(self, args, input_args_names, dtype_list):
        """
        replace the dtype field of args
        """
        dtype_index_list = self.get_args_dtype(input_args_names)
        if not dtype_index_list or len(dtype_index_list) > len(dtype_list):
            self._log.error("replace_args_dtype :: dtype_index_list failed, dtype_index_list:{0},dtype_list:{1}".format(
                dtype_index_list, dtype_list))
            return False

        input_dtype_index = 0
        for index in dtype_index_list:
            args[index] = dtype_list[input_dtype_index]
            input_dtype_index += 1

        return True

    def set_args_dtype(self, args, func, dtype_list):
        """
        Set the dtype field of the use case parameter list
        """
        if not args or not dtype_list:
            self._log.error("set_args_dtype failed for test_arg_list:{0},dtype_list:{1}".format(args, dtype_list))
            return True

        func_input_args_names = func.__code__.co_varnames
        if not func_input_args_names:
            self._log.error("function : {0} args list is None".format(func))
            return True

        return self.replace_args_dtype(args, func_input_args_names, dtype_list)

    def upload_file_ftp(self, upload_type, local_file_path):
        if upload_type not in ("csvs", "cce", "ir", "dump_shape", "logs",):
            self._log.error("upload_file_ftp failed :: not support for upload_type:{0}".format(upload_type))
            return None

        today = str(datetime.date.today())
        ftp = ftpHandle(self._log)
        if not ftp.ftp_login():
            self._log.error("upload_file_ftp failed for ftp_login")
            return None

        remote_path = os.path.join("/auto_tensor", upload_type)
        if not ftp.ftp_mkdir(remote_path, today):
            self._log.error("upload_file_ftp failed for ftp_mkdir,remote_path:{0},today:{1}".format(remote_path, today))
            ftp.ftp_close()
            return None
        remote_path = os.path.join(remote_path, today)
        remote_file_name = str(local_file_path).split("/")[-1]

        if not ftp.ftp_upload_file(remote_path, remote_file_name, local_file_path):
            self._log.error(
                "upload_file_ftp failed for ftp_upload_file,remote_path:{0},today:{1},local_file_path:{2}".format(
                    remote_path, today, local_file_path))
            ftp.ftp_close()
            return None
        ftp_url = "ftp://{host}/{path}".format(host=ftp.host, path=os.path.join(remote_path, remote_file_name))

        ftp.ftp_close()
        return ftp_url

    def collect_ir(self, kernel_name):
        if not os.path.exists(kernel_name):
            self._log.warning("not exist ir directory for :{kernel_name}".format(kernel_name=kernel_name))
            return None

        file_name = kernel_name + ".tar.gz"
        with tarfile.open(file_name, "w:gz") as tar:
            tar.add(kernel_name, arcname=os.path.basename(kernel_name))
        return file_name

    def collect_cce(self, kernel_name):
        file_name = kernel_name + ".cce"
        if not os.path.exists(file_name):
            self._log.warning("not exist cce file for :{file_name}".format(file_name=file_name))
            return None
        return file_name

    def collect_data(self, input, output, cce_file_name, ir_file_name, arg, kernel_name):
        ret_val = True
        # dump input and output
        dump_file_list = self.data_dump(input, output, arg)
        self._log.warning("dump input and output as follow:")
        if os.environ.get("FTP_HOST"):
            for dump_file in dump_file_list:
                ftp_url = self.upload_file_ftp("dump_shape", dump_file)
                if not ftp_url:
                    self._log.error("upload_file_ftp failed for dump_file : {0}".format(dump_file))
                    ret_val = False
                else:
                    self._log.warning("dump_file ftp_url : {0}".format(ftp_url))
            # dump ir
            if not ir_file_name:
                self._log.error("collect_ir failed")
                ret_val = False
            else:
                ftp_url = self.upload_file_ftp("ir", ir_file_name)
                if not ftp_url:
                    self._log.error("upload_file_ftp failed for ir_file_name : {0}".format(ir_file_name))
                    ret_val = False
                else:
                    self._log.warning("ir ftp_url : {0}".format(ftp_url))

            # dump cce
            if not cce_file_name:
                self._log.error("collect_cce failed")
                ret_val = False
            else:
                ftp_url = self.upload_file_ftp("cce", cce_file_name)
                if not ftp_url:
                    self._log.error("upload_file_ftp failed for cce_file_name : {0}".format(cce_file_name))
                    ret_val = False
                else:
                    self._log.warning("cce ftp_url : {0}".format(ftp_url))
        else:
            case_failed_save_path = '/' + '/'.join(os.path.abspath(self.casename).split('/')[1:-1])
            self._log.warning("The input output data of failed use case log have been saved to the path : {0}/data/{1}"
                              .format(case_failed_save_path, kernel_name))
            self._log.warning("The ir data of failed use case log have been saved to the path : {0}/{1}"
                              .format(case_failed_save_path, ir_file_name))
            self._log.warning("The cce data of failed use case log have been saved to the path : {0}/{1}"
                              .format(case_failed_save_path, cce_file_name))
        return ret_val

    def data_dump(self, input, output, arg):
        dump_file_list = []
        operator_name = str(arg[1]).split("_run")[0].split()[-1]
        data_dir = "./data/{0}/".format(operator_name)
        os.popen("mkdir -p %s" % data_dir)
        time.sleep(1)
        if not isinstance(input, list) and not isinstance(input, tuple):
            input = [input]
        if not isinstance(output, list) and not isinstance(output, tuple):
            output = [output]
        data_dict = {"input": input, "output": output}
        for kays in data_dict.keys():
            for index, i in enumerate(data_dict[kays]):
                seq = [operator_name, kays, str(index + 1)] + list(map(str, arg[2])) + [".t"]
                dump_file_name = "_".join(seq).replace("[", "").replace("]", "").replace(",", "-") \
                    .replace(" ", "").replace("(", "").replace(")", "").replace("_.", ".")
                dump_file_name += str(time.time())
                dump_file = os.path.join(data_dir, dump_file_name)
                dump_file_list.append(dump_file)
                tensorio.dump_tensor(i, dump_file)
        return dump_file_list


def get_rtol_atol(op_name, dtype, rtol=5e-03, atol=5e-03):
    run_mode = os.environ.get('RUNTIME_MODE')
    if run_mode in ("rpc_cloud", "air_cloud"):
        if dtype == "float16":
            rtol = atol = 1e-03
        else:
            rtol = atol = 1e-04
    return rtol, atol


def get_splitted_cases(cases, split_nums, split_idx):
    if not isinstance(cases, (list, tuple)):
        raise TypeError("Argument cases must be of type list or tuple.")
    if not isinstance(split_nums, int) or not isinstance(split_idx, int):
        raise TypeError("Arguments split_nums and split_idx must be of type int.")
    if split_nums <= 0 or split_idx < 0 or split_idx >= split_nums:
        raise ValueError("Argument split_nums must > 0, split_idx must be in range [0, split_nums)")

    cases = list(cases)
    all_cases = len(cases)
    fragment = (all_cases + split_nums - 1) // split_nums

    start_idx = split_idx * fragment
    if start_idx >= all_cases:
        return []

    end_idx = start_idx + fragment
    if end_idx > all_cases:
        end_idx = all_cases

    return cases[start_idx:end_idx]
