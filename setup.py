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

import os
from setuptools import setup

version = '1.2.0'

package_data = {
    'akg': [
        '*.so*',
        '*.cuh',
        'lib/*.so*',
        'config/*',
        'include/*',
        'include/*/*',
        'include/*/*/*',
        'include/*/*/*/*'
    ]
}


def find_files(where=['.']):
    """
    Return a package list

    'where' is the root directory list
    """
    dirs = [path.replace(os.path.sep, '.') for path in where]
    for selected_root in where:
        for root, all_dirs, files in os.walk(selected_root, followlinks=True):
            for dir in all_dirs:
                full_path = os.path.join(root, dir)
                package = full_path.replace(os.path.sep, '.')
                if '.' in dir:
                    continue
                dirs.append(package)

    packages = []
    for dir in dirs:
        if dir.endswith("__pycache__"):
            continue
        elif dir.startswith("build."):
            packages.append(dir[6:])
        else:
            packages.append(dir)
    return packages


setup(
    name='akg',
    version=version,
    author='The MindSpore Authors',
    author_email='contact@mindspore.cn',
    url='https://www.mindspore.cn/',
    download_url='https://gitee.com/mindspore/akg/tags',
    project_urls={
        'Sources': 'https://gitee.com/mindspore/akg',
        'Issue Tracker': 'https://gitee.com/mindspore/akg/issues',
    },
    description="An optimizer for operators in Deep Learning Networks, which provides the ability to automatically "
                "fuse ops with specific patterns.",
    license='Apache 2.0',
    package_data=package_data,
    packages=find_files(['build/akg']),
    package_dir={'akg': 'build/akg'},
    include_package_data=True,
    install_requires=[
        'scipy >= 1.5.2',
        'numpy >= 1.17.0',
        'decorator >= 4.4.0'
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
    ]
)
