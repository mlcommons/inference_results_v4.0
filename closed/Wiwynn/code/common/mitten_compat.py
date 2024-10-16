#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class ArgDiscarder:
    """Mitten TensorRTBuilder Ops were designed to be used as Python Mixins, which must claim arguments and forward
    unused ones to super. However, the old API passed around large dicts of config arguments, which contained extraneous
    information (i.e. GenerateEngines would also be given harness run arguments, which would be unused).
    When used with mixins, extraneous arguments will throw an error if object.__init__ is called with unexpected
    arguments.

    This class is a mixin which stops the forwarding of *args and **kwargs.

    DeprecationWarning: Will be removed when the codebase is fully refactored.
    """

    def __init__(self, *args, **kwargs):
        pass
