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

"""Config space"""
from abc import ABCMeta, abstractmethod
from typing import NamedTuple, List
import random
import numpy as np


class ConfigEntity:
    """General config entity"""

    def __init__(self, input_id: int, input_space: NamedTuple):
        self.__input = input_space
        self.__input_id = input_id
        self.__input_type = type(input_space)

    def __len__(self):
        return len(self.__input)

    def __str__(self):
        return str(self.__input_id) + ': ' + str(self.__input)

    def __repr__(self):
        return str(self)

    @property
    def input_id(self):
        return self.__input_id

    @property
    def input_type(self):
        return self.__input_type

    @property
    def input(self):
        return self.__input

    @property
    def feature(self):
        return self.__input


class ConfigSpace(metaclass=ABCMeta):
    """Searching space of configs"""

    def __init__(self, input_type):
        self._input_type = input_type
        self._dim_names = getattr(self._input_type, '_fields')

        self._configs = []  # List[ConfigEntity]

    @abstractmethod
    def reset_fetch(self):
        pass

    @abstractmethod
    def has_next(self) -> bool:
        pass

    @abstractmethod
    def fetch_index(self) -> int:
        """fetch a random index of config"""

    @abstractmethod
    def fetch_config(self) -> ConfigEntity:
        """fetch a random config"""

    @abstractmethod
    def random_walk(self, p: int) -> int:
        """find a neighbor hood of the p-th ConfigEntity, which only
            differs with p in at most one dimension"""

    def get(self, idx: int) -> ConfigEntity:
        """get the `idx`-th config of the space"""
        return self._configs[idx]

    @property
    def configs(self):
        return self._configs

    @property
    def dim_names(self):
        return self._dim_names

    @property
    def input_type(self):
        return self._input_type

    @property
    # @abstractmethod
    def length(self):
        return len(self.configs)


class ConfigTrie:
    """Trie node for config entities"""

    def __init__(self):
        self.ch = dict()

    def add(self, config: ConfigEntity, last_dim: int):
        """add a ConfigEntity"""
        cur = self
        for i, x in enumerate(config.input):
            if i == last_dim:
                continue
            if x not in cur.ch:
                cur.ch[x] = ConfigTrie()
            if not isinstance(cur.ch, dict):
                raise TypeError('none-leaf node should have a dict of childs')
            cur = cur.ch[x]

        if not isinstance(cur.ch, list):
            cur.ch = []
        cur.ch.append(config.input_id)

    def fetch_random(self, config: ConfigEntity, last_dim: int) -> int:
        """randomly fetch the index of a ConfigEntity the same with `config` except for the `last_dim`-th dimension"""
        cur = self
        for i, x in enumerate(config.input):
            if i == last_dim:
                continue
            if not isinstance(cur.ch, dict):
                raise TypeError('none leaf node should have a dict of childs')
            if x not in cur.ch:
                raise RuntimeError('no element found')
            cur = cur.ch[x]
        if not cur.ch:
            raise RuntimeError('no element found')
        if len(cur.ch) == 1:
            return cur.ch[0]
        idx = config.input_id
        while idx == config.input_id:
            idx = random.choice(cur.ch)
        return idx


class ListConfigSpace(ConfigSpace):
    """Searching space of configs, which stores all possible configs in a list"""

    def __init__(self, input_type):
        super(ListConfigSpace, self).__init__(input_type)

        self.__config_tries = [ConfigTrie() for _ in range(len(self._dim_names))]
        self.__fetch_pool = []

    def reset_fetch(self):
        """reset fetch state"""
        self.__fetch_pool = [i for i in range(len(self._configs))]

    def fetch_scope(self, start, end):
        self.__fetch_pool = [i for i in range(start, end)]

    def has_next(self) -> bool:
        return len(self.__fetch_pool) > 0

    def fetch_index(self) -> int:
        """fetch a random index of config"""
        idx = np.random.randint(len(self.__fetch_pool))
        ret = self.__fetch_pool[idx]
        self.__fetch_pool[idx] = self.__fetch_pool[-1]
        self.__fetch_pool.pop()
        return ret

    def fetch_next_index(self) -> int:
        """fetch next index of config"""
        idx = len(self.__fetch_pool) - 1 + self.__fetch_pool[0]
        self.__fetch_pool.pop()
        return idx

    def fetch_config(self) -> ConfigEntity:
        """fetch a random config"""
        return self.get(self.fetch_index())

    def add(self, input_space: NamedTuple):
        """add a new config to space"""
        if not isinstance(input_space, self._input_type):
            raise TypeError('invalid config input space type, got {} expected {}'.format(type(input_space),
                                                                                         self._input_type))
        config = ConfigEntity(len(self._configs), input_space)
        self.__fetch_pool.append(len(self._configs))
        for i in range(len(self._dim_names)):
            self.__config_tries[i].add(config, i)
        self._configs.append(config)

    def random_walk(self, p: int) -> int:
        """find a neighbor hood of the p-th ConfigEntity, which only differs with p in at most one dimension"""
        dim = np.random.randint(len(self._dim_names))
        return self.__config_tries[dim].fetch_random(self._configs[p], dim)

    @property
    def length(self):
        return len(self._configs)

    @classmethod
    def from_list(cls, configs: List[NamedTuple]):
        if not isinstance(configs, list):
            raise TypeError('configs must be of list type, got %s' % type(configs))
        if not configs:
            raise ValueError('configs must be non-empty')
        space = cls(type(configs[0]))
        for config in configs:
            space.add(config)
        return space
