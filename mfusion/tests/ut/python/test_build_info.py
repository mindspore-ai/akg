# Copyright 2026 Huawei Technologies Co., Ltd
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

"""UT for package build metadata."""

from pathlib import Path

import mfusion


def test_build_info_is_exposed():
    """Verify installed package metadata exposes version and commit id."""
    build_info = mfusion.get_build_info()
    commit_id_file = Path(mfusion.__file__).with_name(".commit_id")

    assert build_info["version"] == mfusion.__version__
    assert build_info["commit_id"] == mfusion.__commit_id__
    assert isinstance(mfusion.__version__, str) and mfusion.__version__
    assert isinstance(mfusion.__commit_id__, str) and mfusion.__commit_id__
    assert commit_id_file.read_text(encoding="utf-8").strip() == mfusion.__commit_id__
