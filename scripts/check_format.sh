#!/bin/bash
# Copyright 2021 The QHBM Library Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

poetry run yapf --diff --recursive qhbmlib/
retval=$?
if [ "$retval" == 0 ]
then
  echo "Success: library files are formatted correctly."
else
  echo "Please run `poetry run yapf --in-place --recursive qhbmlib/` and try again."
  exit 1
fi
poetry run yapf --diff --recursive tests/
retval=$?
if [ "$retval" == 0 ]
then
  echo "Success: test files are formatted correctly."
else
  echo "Please run `poetry run yapf --in-place --recursive tests/` and try again."
  exit 1
fi