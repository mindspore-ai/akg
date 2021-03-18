# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
# 2020.12.17 - Fit for AKG tuning model.
"""XGBoost as cost model"""

import time
import logging
import numpy as np
import xgboost as xgb

logger = logging.getLogger('fuzz.tune.autotuning.xgb_cost_model')

class XgbCostModel:
    """Cost model to predict the speed of a config"""

    def __init__(self, space):
        self.space = space

        self.xgb_params = {
            'max_depth': 6,
            'gamma': 0.0001,
            'min_child_weight': 1,

            'subsample': 1.0,

            'eta': 0.3,
            'lambda': 1.00,
            'alpha': 0,

            'objective': 'rank:pairwise',
            'verbosity': 0
        }
        self._sample_size = 0
        self.bst = None
        self.log_interval = 25

    def fit(self, xs, ys, plan_size):
        """XGBoost fitting """

        tic = time.time()

        x_train = self._get_feature(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        valid_index = y_train > 1e-6
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self._sample_size = len(x_train)

        self.bst = xgb.train(self.xgb_params, dtrain,
                             num_boost_round=8000,
                             callbacks=[custom_callback(
                                 stopping_rounds=20,
                                 metric='tr-a-recall@%d' % plan_size,
                                 evals=[(dtrain, 'tr')],
                                 fevals=[
                                     xgb_average_recalln_curve_score(plan_size),
                                 ],
                                 verbose_eval=self.log_interval)])

        logger.debug("XGB train: %.2f\tobs: %d\terror: %d\tn_cache: %d",
                     time.time() - tic, len(xs),
                     len(xs) - np.sum(valid_index),
                     0)

    def predict(self, xs, output_margin=False):
        feas = self._get_feature(xs)
        dtest = xgb.DMatrix(feas)

        return self.bst.predict(dtest, output_margin=output_margin)

    def _get_feature(self, indexes):
        ret = np.empty((len(indexes), len(self.space.get(0).feature)), dtype=np.float32)
        for i, ii in enumerate(indexes):
            ret[i, :] = np.array(self.space.get(ii).feature, dtype=np.float32)

        return ret

    def reset_space(self, space):
        self.space = space


def custom_callback(stopping_rounds, metric, fevals, evals=(), log_file=None,
                    verbose_eval=25):
    """callback function for xgboost to support multiple custom evaluation functions"""
    from xgboost.core import EarlyStopException
    from xgboost.callback import _fmt_metric
    from xgboost.training import aggcv

    best_state = {}
    metric_shortname = metric.split("-")[1]

    def init(env):
        """internal function"""
        bst = env.model

        best_state['iteration'] = 0
        best_state['score'] = float('-inf')
        best_state['msg'] = ''

        if bst is not None:
            if bst.attr('best_score') is None:
                bst.set_attr(best_iteration=str(best_state['iteration']))
                bst.set_attr(best_score=str(best_state['score']))
            else:
                best_state['score'] = float(bst.attr('best_score'))
                best_state['iteration'] = int(bst.attr('best_iteration'))
                best_state['msg'] = bst.attr('best_msg')

    def callback(env):
        """internal function"""
        if not best_state:
            init(env)

        bst = env.model
        i = env.iteration
        cvfolds = env.cvfolds

        res_dict = {}

        ##### evaluation #####
        if cvfolds is not None:
            for feval in fevals:
                tmp = aggcv([f.eval(i, feval) for f in cvfolds])
                for k, mean, std in tmp:
                    res_dict[k] = [mean, std]
        else:
            for feval in fevals:
                bst_eval = bst.eval_set(evals, i, feval)
                res = [x.split(':') for x in bst_eval.split()]
                for kv in res[1:]:
                    res_dict[kv[0]] = [float(kv[1])]

        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        ##### print eval result #####
        infos = ["XGB iter: %3d" % i]
        for item in eval_res:
            if 'null' in item[0]:
                continue
            infos.append("%s: %.6f" % (item[0], item[1]))

        if not isinstance(verbose_eval, bool) and verbose_eval and i % verbose_eval == 0:
            logger.debug("\t".join(infos))
        if log_file:
            with open(log_file, "a") as fout:
                fout.write("\t".join(infos) + '\n')

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == metric:
                score = item[1]
                break

        best_score = best_state['score']
        best_iteration = best_state['iteration']
        if score and score > best_score:
            msg = '[%d] %s' % (env.iteration, '\t'.join([_fmt_metric(x) for x in eval_res]))
            if bst is not None:
                bst.set_attr(best_score=str(score), best_iteration=str(env.iteration), best_msg=msg)
            best_state['msg'] = msg
            best_state['score'] = score
            best_state['iteration'] = env.iteration
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = best_state['msg']
            if verbose_eval and env.rank == 0:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback


def xgb_average_recalln_curve_score(n):
    """evaluate average recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@%d" % n, np.sum(curve[:n]) / n

    return feval


def recall_curve(trial_ranks, top=None):
    """
    if top is None, f(n) = sum([I(rank[i] < n) for i < n]) / n
    if top is K,    f(n) = sum([I(rank[i] < K) for i < n]) / K

    Parameters
    ----------
    trial_ranks: Array of int
        the rank of i th trial in labels
    top: int or None
        top-n recall

    Returns
    -------
    curve: Array of float
        function values
    """
    if not isinstance(trial_ranks, np.ndarray):
        trial_ranks = np.array(trial_ranks)

    ret = np.zeros(len(trial_ranks))
    if top is None:
        for i in range(len(trial_ranks)):
            ret[i] = np.sum(trial_ranks[:i] <= i) / (i + 1)
    else:
        for i in range(len(trial_ranks)):
            ret[i] = 1.0 * np.sum(trial_ranks[:i] < top) / top
    return ret


def get_rank(values):
    """get rank of items

    Parameters
    ----------
    values: Array

    Returns
    -------
    ranks: Array of int
        the rank of this item in the input (the largest value ranks first)
    """
    tmp = np.argsort(-values)
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(tmp))
    return ranks
