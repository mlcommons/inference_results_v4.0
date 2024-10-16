# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
# experiment config
model_id = "EleutherAI/gpt-j-6b"
dataset_id = "cnn_dailymail"
dataset_config = "3.0.0"
save_dataset_path = "data"
text_column = "article"
summary_column = "highlights"

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
import os
import simplejson as json

# Load dataset from the hub
dataset = load_dataset(dataset_id, name=dataset_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

# Dataset statistics
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

instruction_template = "Summarize the following news article:"

prompt_length = len(tokenizer(instruction_template)["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_length
print(f"Prompt length: {prompt_length}")
print(f"Max input length: {max_sample_length}")

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[text_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_sample_length)
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[summary_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# use 95th percentile as max target length
max_target_length = int(np.percentile(target_lenghts, 95))
print(f"Max target length: {max_target_length}")


def preprocess_function(sample, padding="max_length"):
    # create list of samples
    inputs = []

    # print(type(sample[text_column]))
    # print(len(sample[text_column]))
    # print(len(sample[summary_column]))
    # print(sample[text_column][0])
    # print(sample[summary_column][0])

    for i in range(0, len(sample[text_column])):
        x = dict()
        x["instruction"] = instruction_template
        x["input"] = sample[text_column][i]
        x["output"] = sample[summary_column][i]
        inputs.append(x)
    model_inputs = dict()
    model_inputs["text"] = inputs

    return model_inputs


# process dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))

# save dataset to disk
if not os.path.isdir(save_dataset_path):
    os.makedirs(save_dataset_path)

with open(os.path.join(save_dataset_path, "cnn_train.json"), 'w') as write_f:
    json.dump(tokenized_dataset["train"]["text"], write_f, indent=4, ensure_ascii=False)
with open(os.path.join(save_dataset_path, "cnn_eval.json"), 'w') as write_f:
    json.dump(tokenized_dataset["test"]["text"], write_f, indent=4, ensure_ascii=False)
