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

"""
Log collection function:
import logging plugin, achieve all-round log printing.
You can arbitrarily choose the required log level as
it is divided by the logging module.

such as:
2019-09-09 15:17:15 INFO *.py: func name: line
2019-09-09 14:32:57 WARNING *.py: func name: line
2019-09-09 15:11:41 ERROR *.py: func name: line
"""

import os
import logging
import logging.handlers
from logging.handlers import TimedRotatingFileHandler
import traceback
import random
import string
import sys


class Log(logging.Logger):
    def __init__(self, case_name, case_path):
        super(Log, self).__init__(case_name)
        self.log = logging.getLogger(case_name + ''.join([random.choice(string.digits + string.ascii_letters) for _ in range(8)]))
        self.log.setLevel(logging.DEBUG)
        fmt = '%(levelname)s %(asctime)s - %(filename)s:%(funcName)s:%(lineno)s - %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, datefmt)
        logfile = os.path.join(case_path, '{0}.log'.format(case_name))
        fh = TimedRotatingFileHandler(logfile, when='D', interval=1, backupCount=10)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.log.removeHandler(fh)
        self.log.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.log.removeHandler(ch)
        self.log.addHandler(ch)

    def traceback(self):
        """
        The traceback module prints out the details of the case execution failure.
        """
        self.log.error("There are something error appear.")
        traceback.print_exc()
