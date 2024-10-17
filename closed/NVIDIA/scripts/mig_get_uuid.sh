#!/bin/bash
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

# This script outputs the UUID of a random MIG instance to be used by other scripts.
number_of_instances=$(nvidia-smi -L | grep MIG | wc -l)
random_instance=$(( ( $RANDOM % ${number_of_instances} ) + 1))
instance=$(nvidia-smi -L | grep MIG | sed 's/)//g' | head -1 | tail -1)
instance=${instance#*UUID: }
echo ${instance}
