import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
import copy
import pickle

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-Dataset")

import random
import array
from transformers import LlamaTokenizerFast

class Dataset():
    def __init__(self, model_name=None, total_sample_count=24576, perf_count_override=None, dataset_path=None, device="cpu", model_path=None):
        self.model_name = model_name or "meta-llama/Llama-2-70b-chat-hf"
        self.dataset_path = dataset_path
        self.max_length = 1024
        self.device = device
        self.model_path = model_path

        self.load_tokenizer()
        self.load_processed_dataset()

        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count


    def load_tokenizer(self):
        """ Returns tokenizer """

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warn("Processed pickle file {} not found. Please check that the path is correct".format(self.dataset_path))

        print("Loading dataset...")
        import pandas as pd
        processed_data = pd.read_pickle(self.dataset_path)

        input_tokens = processed_data['tok_input']

        self.input_ids = []
        self.input_lens = []
        self.attention_masks = []

        for ids in input_tokens:
            input_ids = torch.tensor(ids, dtype=torch.int32).view(1,-1).to(self.device)
            attn_mask = torch.ones_like(input_ids)
            self.input_ids.append(input_ids)
            self.attention_masks.append(attn_mask)
            self.input_lens.append(input_ids.shape[-1])


    def save_outputs_to_file(self,query_id_list,output_seq):
        assert query_id_list is not None 
        assert output_seq is not None

        job_id = os.environ["SLURM_JOB_ID"]
        output_path_dir = f"run_outputs/job_{job_id}"
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)

        fname = "q" + "_".join([str(i) for i in query_id_list])
        fname = f"{output_path_dir}/{fname}.pkl"
        with open(fname, mode='wb') as f:
            d = {"query_ids": query_id_list,
                "outputs": output_seq}
            pickle.dump(d, f)

    def post_process(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None):
        """ Postprocesses output prediction """

        #TODO: Create response object in postProcess(?)
        """
        preds = []
        for i in range(out_tokens.shape[0]):
            #pred = out_tokens[i].reshape(-1).cpu().numpy() # Slice up to original input length as below?

            input_len = input_seq_lens[i] if input_seq_lens else 0
            pred = out_tokens[i, input_len:].reshape(-1).cpu().numpy()
            preds.append(pred)
        """
        batch_size, _, _ = out_tokens.size()
        output_seq = []
        for batch_idx in range(batch_size):
            output_seq.append(out_tokens[batch_idx][0].cpu().numpy())

        assert len(query_id_list) == len(output_seq)

        # Save outputs
        job_id = os.environ["SLURM_JOB_ID"]
        output_path_dir = f"run_outputs/job_{job_id}"
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)

        if len(query_id_list) <= 32:
            self.save_outputs_to_file(query_id_list,output_seq)
            # fname = "q" + "_".join([str(i) for i in query_id_list])
            # fname = f"{output_path_dir}/{fname}.pkl"
            # with open(fname, mode='wb') as f:
            #     d = {"query_ids": query_id_list,
            #         "outputs": output_seq}
            #     pickle.dump(d, f)
        # need to batch the saves as names get too long for linux
        else:  
            query_id_list_tmp = []
            output_seq_tmp = []
            # loop over query_id list
            for i in range(len(query_id_list)):
                # append current values
                query_id_list_tmp.append(query_id_list[i])
                output_seq_tmp.append(output_seq[i])
                # every 32 sequence ids dump the files and reset tmp lists
                if i % 32 == 0 and i != 0:
                    self.save_outputs_to_file(query_id_list_tmp,output_seq)
                    # fname = "q" + "_".join([str(i) for i in query_id_list_tmp])
                    # fname = f"{output_path_dir}/{fname}.pkl"
                    # with open(fname, mode='wb') as f:
                    #     d = {"query_ids": query_id_list_tmp,
                    #         "outputs": output_seq_tmp}
                    #     print(f"Saving outputs to {fname}")
                    #     pickle.dump(d, f)
                    query_id_list_tmp = []
                    output_seq_tmp = []

            if len(query_id_list_tmp) != 0:
                self.save_outputs_to_file(query_id_list_tmp,output_seq)
                # fname = "q" + "_".join([str(i) for i in query_id_list_tmp])
                # fname = f"{output_path_dir}/{fname}.pkl"
                # with open(fname, mode='wb') as f:
                #     d = {"query_ids": query_id_list_tmp,
                #         "outputs": output_seq_tmp}
                #     print(f"Saving outputs to {fname}")
                #     pickle.dump(d, f)

        return output_seq
    
    def load_samples_from_ram(self, sample_list):
        pass

    def unload_samples_from_ram(self, sample_list):
        pass

    def __del__(self):
        pass
    
    def get_model_name(engine_dir=None):

        engine_version = tensorrt_llm.runtime.engine.get_engine_version(engine_dir)
        with open(Path(engine_dir) / "config.json", 'r') as f:
            config = json.load(f)

        if engine_version is None:
            return config['builder_config']['name']

        return config['pretrained_config']['architecture']