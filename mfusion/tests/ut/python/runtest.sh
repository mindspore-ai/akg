#!/bin/bash

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)

export PYTHONPATH="${BASEPATH}:${PYTHONPATH}"

pytest -v "${BASEPATH}"
