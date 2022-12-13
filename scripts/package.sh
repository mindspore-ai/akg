#!/bin/bash
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

set -e

BASEPATH=$(cd "$(dirname $0)/../"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
PYTHON=$(which python3)

if [ ! -d ${OUTPUT_PATH} ]; then
    mkdir -pv "${OUTPUT_PATH}"
fi

write_version() {
    if [ ! -e ${BASEPATH}/version.txt ]; then
        version=`git branch | sed -n '/\* /s///p'`
        if [ -z ${version} ]; then
            version='master'
        fi
        echo ${version#r} > ${BASEPATH}/version.txt
    fi
}

write_checksum() {
    cd "${OUTPUT_PATH}"
    PACKAGE_LIST=$(ls akg-*.whl)
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

write_version
cd ${BASEPATH}
${PYTHON} setup.py sdist bdist_wheel

for file in `ls ${BASEPATH}/dist/*.whl`
do
    file_name=$(basename $file)
    prefix=`echo $file_name | cut -d '-' -f 1-2`
    CUR_ARCH=`arch`
    PY_VERSION=`python3 --version`
    PY_TAGS=""
    if [[ $PY_VERSION == *3.7* ]]; then
        PY_TAGS="cp37-cp37m"
    elif [[ $PY_VERSION == *3.8* ]]; then
        PY_TAGS="cp38-cp38"
    elif [[ $PY_VERSION == *3.9* ]]; then
        PY_TAGS="cp39-cp39"
    else
        echo "Error: Could not find Python between 3.7 and 3.9"
        exit 1
    fi
    new_file_name="${prefix}-${PY_TAGS}-linux_${CUR_ARCH}.whl"
    mv $file ${BASEPATH}/dist/${new_file_name}
done

mv ${BASEPATH}/dist/*.whl ${OUTPUT_PATH}
write_checksum
echo "------Successfully created akg package------"
