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

"""Tuner for finding best config for operators"""
import logging
import time
import json
import os
import numpy as np
from .xgb_cost_model import XgbCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer
from .space import ConfigSpace
from .runner import KernelRunner

logger = logging.getLogger('fuzz.tune.autotuning.tuner')


class Tuner:
    """Basic tuner class

    Parameters
    ----------
    runner: KernelRunner
        This is for run kernels in physical device
    config_space: ConfigSpace
        The space of configs
    n_parallel: int
        How many kernels are processed in a turn
    """

    def __init__(self, runner: KernelRunner, index_table: list, config_space: ConfigSpace, n_parallel: int = 1):
        self._runner = runner
        self._index_table = index_table
        self._space = config_space
        self._n_parallel = n_parallel

        # trial plan
        self._trials = []
        self._trial_pt = 0
        self._visited = set()

        # observed samples
        self._xs = []
        self._ys = []

        # keep the current best
        self._best_config = None  # type: ConfigEntity
        self._best_time = np.inf
        self._best_iter = 0
        self._tuning_time = 0.0
        self._original_time = np.inf

    @property
    def best_config(self):
        return self._best_config

    @property
    def best_time(self):
        return self._best_time

    @property
    def best_iter(self):
        return self._best_iter

    @property
    def original_time(self):
        return self._original_time

    @property
    def xs(self):
        return self._xs

    @property
    def ys(self):
        return self._ys

    def info(self):
        print('space size:', self._space.length)
        print('best config:', self._best_config)
        print('best time:', self._best_time)
        print('best_iter:', self._best_iter)
        print('tuning time:', self._tuning_time, 'secs')

    def next_batch(self, batch_size: int, is_add_visited=True):
        """extract next batch"""
        ret = []
        counter = 0
        if not is_add_visited:
            return [self._space.get(index) for index in range(min(batch_size, self._space.length))]
        while counter < batch_size and self._space.has_next():
            index = 0
            while self._trial_pt < len(self._trials):
                index = self._trials[self._trial_pt]
                if index not in self._visited:
                    break
                self._trial_pt += 1

            if self._trial_pt >= len(self._trials):
                # if the trial list is empty choose randomly
                index = self._space.fetch_index()

            ret.append(self._space.get(index))
            self._visited.add(index)

            counter += 1
        return ret

    def export_configs(self, configs: list, output_file: str, append: bool = True, desc=""):
        """export configs"""
        mode = "a" if append else "w"
        with open(output_file, mode) as f:
            for x, y in configs:
                f.write("{} | {} | {}\n".format(desc, json.dumps(x._asdict()), y))

    def export_dim_configs(self, configs, output_file: str, append: bool = True, key=""):
        """export dim configs"""
        mode = "a" if append else "w"
        data = {}
        try:
            if os.path.isfile(output_file):
                with open(output_file, 'r') as f:
                    data = json.load(f)
        except IOError as e:
            logger.debug("get dim info from [%s] failed: %s", output_file, str(e))
        with open(output_file, mode) as f:
            import re
            data[key] = configs
            s = json.dumps(data, sort_keys=True)
            s = re.sub(r',\s*"', ',\n"', s)
            s = '{\n' + s[1:-1] + '\n}'
            f.write(s)

    def load_configs(self, input_file: str):
        """load configs"""
        configs = []
        file_path = os.path.realpath(input_file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    x, y, _ = line.split('|')
                    configs.append((self._space.input_type(**json.loads(x)), np.float64(y)))
        return configs

    def tune(self, least_try_times: int, output_file: str = None):
        """grid search all configs"""
        i = 0
        while i < least_try_times:
            if not self._space.has_next():
                break
            configs = self.next_batch(min(self._n_parallel, least_try_times - i))
            run_times = self._runner.run(configs, self._best_time)
            results = []
            for idx, conf in enumerate(configs):
                results.append((conf.input_id, run_times[idx]))
                # keep best config
                if self.best_time < run_times[idx]:
                    self._best_time = run_times[idx]
                    self._best_iter = i + idx
                    self._best_config = conf

            i += len(results)

            # update
            for res in results:
                self._xs.append(res[0])
                self._ys.append(res[1])
            if output_file:
                configs = [(self._space.get(res[0]).input, res[1]) for res in results]
                self.export_configs(configs, output_file)
        return run_times


class ModelBasedTuner(Tuner):
    """Model based tuner
    This tuner will fit a cost model and use an optimizer to find the maximums of the cost model as next trials

    Parameters
    ----------
    plan_size: int
        Tuner will re-fit model per `plan_size` new measure samples
    pre_model: CostModel
        The cost model that predicts the speed of a config (IR)
    """

    def __init__(self, runner, index_table, config_space, n_parallel=1, plan_size=32, pre_model=None):
        super(ModelBasedTuner, self).__init__(runner, index_table, config_space, n_parallel)
        self.__plan_size = plan_size

        if pre_model is not None:
            self.__cost_model = pre_model
            self.__cost_model.reset_space(self._space)
        else:
            self.__cost_model = XgbCostModel(self._space)

        self.__model_optimizer = SimulatedAnnealingOptimizer(self._space)
        self.__train_ct = 0

        self.__is_auto_set_dim = True

        # time to leave
        self.__ttl = None
        self.__least_try_times = None
        self.__early_stopping = None

        self.__model_run_time = 0.0

    def info(self):
        super(ModelBasedTuner, self).info()
        print('model run time:', self.__model_run_time, 'secs')

    def tune(self, least_try_times: int, output_file: str = None):
        early_stopping = least_try_times
        self.__least_try_times = least_try_times
        self.__early_stopping = early_stopping

        old_level = logger.level
        i = 0
        error_ct = 0

        tuning_start = time.time()
        while (i < self._space.length and (i < least_try_times
                                           or (self._best_time > self._original_time - 0.9
                                               and i < least_try_times * 3))):
            if not self._space.has_next():
                break
            iter_start = time.time()
            if not self.__is_auto_set_dim:
                configs = self.next_batch(min(self._n_parallel, self._space.length - i))
            else:
                configs = self.next_batch(min(self._n_parallel, self._space.length - i), False)

            logger.debug('--indexes: %s', str([x.input_id for x in configs]))

            run_times = self._runner.run(configs, self._best_time, self.__is_auto_set_dim)
            if self.__is_auto_set_dim:
                from operator import add
                from functools import reduce
                self._original_time = reduce(add, run_times) / len(run_times)
                self._best_time = self._original_time
                self._best_iter = -1
                self._best_config = None
                run_times = None
                self.__is_auto_set_dim = False
                continue

            results = []
            for idx, conf in enumerate(configs):
                results.append((conf.input_id, run_times[idx]))
                # keep best config
                if self._best_time > run_times[idx]:
                    self._best_time = run_times[idx]
                    self._best_iter = i + idx
                    self._best_config = conf

            i += len(results)
            self.__ttl = min(early_stopping + self.best_iter, self._space.length) - i

            start = time.time()
            # update
            for res in results:
                self._xs.append(res[0])
                self._ys.append(res[1])
            if output_file:
                configs = [(self._space.get(res[0]).input, res[1]) for res in results]
                desc = str(self._runner.op_desc)
                self.export_configs(configs, output_file, desc=desc)

            # if we have enough new training samples
            if len(self._xs) >= self.__plan_size * (self.__train_ct + 1):
                self.__cost_model.fit(self._xs, self._ys, self.__plan_size)
                best_configs = self.__model_optimizer.find_best(
                    self.__cost_model, self.__plan_size, self._visited)

                self._trials = best_configs
                self._trial_pt = 0
                self.__train_ct += 1

            end = time.time()
            logger.debug('model running time: %f seconds', end - start)
            self.__model_run_time += end - start

            iter_end = time.time()
            logger.debug('iter time: %f seconds', iter_end - iter_start)

            if self._best_iter > 0 and i >= self.best_iter + early_stopping:
                logger.debug('Early stopped. Best iter: %d', self._best_iter)
                return

            if error_ct > 150:
                logging.warning('Too many errors happen in the tuning. Now is in debug mode')
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        self._tuning_time += time.time() - tuning_start
