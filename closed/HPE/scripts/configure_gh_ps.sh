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

# This script configures G+H's power-sloshing feature using nvidia-smi
# Arguments:
#     $1: 0 to disable power-sloshing / 1 to enable power-sloshing

TURNON=$1

if [[ $TURNON -eq 1 ]]; then
echo "Enabling G+H Power Sloshing..."
sudo nvidia-smi --power-limit 1000 --scope 0
sudo nvidia-smi --power-limit 1000 --scope 1
elif [[ $TURNON -eq 0 ]]; then
echo "Resetting G+H Power Configuration..."
sudo nvidia-smi --power-limit 700 --scope 0
sudo nvidia-smi --power-limit 1000 --scope 1
fi
