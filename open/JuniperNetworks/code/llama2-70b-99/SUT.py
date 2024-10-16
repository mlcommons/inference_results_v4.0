import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

import pickle
import time
import tqdm
import queue
import threading

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset

import argparse
from argparse import ArgumentParser
import ast
import csv
from pathlib import Path
import os
import sys
import asyncio

import numpy as np
import torch
from utils import read_model_name

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

from tensorrt_llm.engine import AsyncLLMEngine
from tensorrt_llm.runtime import SamplingConfig
import copy

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp
    
os.environ["CUDA_VISIBLE_DEVICES"] = str(tensorrt_llm.mpi_rank()%8)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-SUT")
lock = threading.Lock()
qlock = threading.Lock()

class FirstTokenStreamer(BaseStreamer):
    """ Streams first tokens to a 'holder' """

    def __init__(self, first_token, tokens_cache=[], is_first_token=True, response_ids=[] ):
        """ Response ids added to 'sign' the first token"""

        self.first_token = first_token # Queue for first token
        self.is_first_token = is_first_token

        # Cache for subsequent generated tokens
        self.tokens_cache = tokens_cache

        self.response_ids = response_ids

        self.is_prompt = True # The first tokens sent to the streamer are actually the input prompts

    def put(self, value):
        """ Caches the tokens as they're generated. Assumes bs=1 """

        # Prompts are streamed first so we need to skip the first time value that arrives
        if self.is_prompt:
            self.is_prompt = False
            return

        value = value.item()
        if self.is_first_token:

            # Add generated first token together with its query response_id to first tokens queue
            self.first_token.put((value, self.response_ids[0]))

            self.is_first_token = False
            return

        self.tokens_cache.append(value)


    def end(self):
        pass

    def get_out_tokens(self):
        return self.tokens_cache


class SUT():
    def __init__(self,
                 model_path=None,
                 dtype="bfloat16",
                 batch_size=None,
                 total_sample_count=24576,
                 dataset_path=None,
                 use_cached_outputs=False,
                 workers=1,
                 engine_dir=None):

        log.info(f"Num workers: {workers}")
        self.runtime_rank = tensorrt_llm.mpi_rank()
        self.device = torch.device("cuda:"+str(self.runtime_rank%8))
        self.model_path = model_path
        self.engine_dir = engine_dir
        self.num_workers = workers
        self.access = 0

        if not batch_size:
            batch_size = 32
        self.batch_size = batch_size

        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path,
                                   dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count,
                                   device=self.device,
                                   model_path=model_path)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count, self.data_object.perf_count,
                                   self.data_object.load_samples_from_ram, self.data_object.unload_samples_from_ram)

        self.load_model()
        
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0

        log.info("Init done")

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries, daemon=True)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()
        self.query_queue.put(None)
        

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """

        runner_kwargs = dict(engine_dir=self.engine_dir,
                            rank=self.runtime_rank,
                            free_gpu_memory_fraction=0.85,
                            max_output_len=1024)

        model = ModelRunnerCpp.from_dir(**runner_kwargs)

        while True:
           
            tik1 = time.time()
            input_ids_tensor = []
            input_masks_tensor = []
            input_len = []
            
            qitem = self.query_queue.get()
            if qitem is None:
                log.info("Queue empty, no samples left")
                break

            query_ids = [q.index for q in qitem]
            for q in qitem:
                input_ids_tensor.append(self.data_object.input_ids[q.index].squeeze().to(self.device))
                attn_mask = self.data_object.attention_masks[q.index].squeeze()
                if attn_mask.shape[0] != attn_mask.sum():
                    log.info("Attention mask was not all ones for query {q.index}")
                    raise Exception(f"Attention mask was not all ones for query {q.index}")
                input_len.append(self.data_object.input_lens[q.index])

            tik2 = time.time()

            with torch.inference_mode():
                # SUT
                pred_output_tokens = model.generate(
                    input_ids_tensor,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    return_dict=False,
                    do_sample=False,
                    early_stopping=True, 
                    max_new_tokens=1024,
                    min_length=10,
                    min_new_tokens=20,
                    temperature=0.8,
                    top_k=0,
                    top_p=0.9,
                    length_penalty=1.0,
                    repetition_penalty=1.1,
                    presence_penalty=0.0)
            
            tik3 = time.time()

            if self.runtime_rank == 0:
                processed_output = self.data_object.post_process(pred_output_tokens,
                                                                input_seq_lens=input_len,
                                                                query_id_list=query_ids)
                for i in range(len(qitem)):
                    n_tokens = processed_output[i].shape[0]
                    response_array = array.array("B", processed_output[i].tobytes())
                    bi = response_array.buffer_info()
                    response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                    lg.QuerySamplesComplete(response)

                tok = time.time()

                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")
                else:
                    log.info(f"\tLoaded from cache: {_p}")
                log.info("Query Processed")

    def load_model(self):
        
        log.info("Load Model Called")
        self.model_name = read_model_name(self.engine_dir)
        self.tokenizer = self.data_object.load_tokenizer()
        log.info("Loaded tokenizer")
        self.end_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
     

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl


    def predict(self,**kwargs):
        raise NotImplementedError


    def issue_queries(self, query_samples):
        log.info("issue_queries called")
        """ Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        list_prompts_tokens = []
        list_prompts_attn_masks = []

        print(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        print(f"IssueQuery done")


    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
	def __init__(self, 
				 model_path=None, 
				 dtype="bfloat16", 
				 total_sample_count=24576, 
				 dataset_path=None, 
				 workers=1, 
				 engine_dir=None,
				 batch_size=8):
		
		super().__init__(model_path=model_path, dtype=dtype, total_sample_count=total_sample_count, 
							dataset_path=dataset_path, workers=workers, engine_dir=engine_dir, batch_size=batch_size)
		self.first_token_queue = queue.Queue()
		self.batch_size = batch_size
	def start(self):
		# Create worker threads
		for j in range(self.num_workers):
			worker = threading.Thread(target=self.process_queries)
			worker.start()
			self.worker_threads[j] = worker

		# Create first token response thread
		self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
		self.ft_response_thread.start()

	def process_first_tokens(self):

		while True:
			first_token_item = self.first_token_queue.get()

			if first_token_item is None:
				log.info("Exiting First token response thread")
				break

			first_tokens, response_id = first_token_item

			response_data = array.array("B", np.array(first_tokens, np.float32).tobytes())
			bi = response_data.buffer_info()
			response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
			lg.FirstTokenComplete(response)

	def process_queries(self):
		"""Processor of the queued queries. User may choose to add batching logic """
		
		runner_kwargs = dict(engine_dir=self.engine_dir,
							rank=self.runtime_rank,
							free_gpu_memory_fraction=0.85)
		model = ModelRunnerCpp.from_dir(**runner_kwargs)
		count = 0

		input_ids_tensor = []
		input_len = []
		qitems = []
		
		while True:
			count+=1
			
			qitem = self.query_queue.get()
			if qitem is None:
				log.info("Queue empty, no queries left")
				break
			
			tik1 = time.time()

			input_ids_tensor.append(self.data_object.input_ids[qitem.index].squeeze().to(self.device))
			input_len.append(self.data_object.input_lens[qitem.index])
			qitems.append(qitem)

			tik2 = time.time()

			with torch.inference_mode():
				output_tokens = model.generate(
						input_ids_tensor,
						end_id=self.end_id,
						pad_id=self.pad_id,
						return_dict=False,
						do_sample=False,
						early_stopping=True, 
						max_new_tokens=1024,
						min_length=30,
						min_new_tokens=20,
						temperature=0.8,
						top_k=0,
						top_p=0.9,
						length_penalty=1.3,
						repetition_penalty=1.2,
						presence_penalty=0.0)
				torch.cuda.synchronize()
				
				tik3 = time.time()

				if self.runtime_rank == 0:
					n_tokens = output_tokens[0].shape[0]
					response_array = array.array("B", np.array(output_tokens[0].cpu(), np.int32).tobytes())
					bi = response_array.buffer_info()
					response = [lg.QuerySampleResponse(
						qitem.id, bi[0], bi[1], n_tokens)]
					lg.QuerySamplesComplete(response)
		
					tok = time.time()
				
					self.sample_counter += len(input_ids_tensor)
					log.info(f"Samples run: {self.sample_counter}")
					log.info(f"\tBatchMaker time: {tik2 - tik1}")
					log.info(f"\tInference time: {tik3 - tik2}")
					log.info(f"\tPostprocess time: {tok - tik3}")
					log.info(f"\t==== Total time: {tok - tik1}")

			input_ids_tensor = []
			input_len = []
			qitems = []

	def issue_queries(self, query_samples):
		self.query_queue.put(query_samples[0])

	def stop(self):
		for _ in range(self.num_workers):
			self.query_queue.put(None)

		for worker in self.worker_threads:
			worker.join()

		self.first_token_queue.put(None)
		self.ft_response_thread.join()