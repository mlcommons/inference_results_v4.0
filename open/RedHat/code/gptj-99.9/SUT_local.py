import os
import sys
import time
import re
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

import pickle
import time
import threading
import tqdm
import queue

from concurrent.futures.thread import ThreadPoolExecutor
import requests
from urllib3.exceptions import InsecureRequestWarning
import json

from inference import GrpcClient

from vllm import LLM, SamplingParams

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GPT-J-SUT")

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": 4,
}



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
                 api_server=None,
                 api_model_name=None,
                 grpc=False,
                 batch_grpc=False,
                 vllm=False,
                 dtype="bfloat16",
                 device="cpu",
                 batch_size=None,
                 total_sample_count=13368,
                 dataset_path=None,
                 use_cached_outputs=False,  # Set this to True *only for test accuracy runs* in case your prior session was killed partway through
                 workers=1):

        self.model_path = model_path or "EleutherAI/gpt-j-6B"
        self.api_server = api_server
        self.api_model_name = api_model_name
        self.grpc = grpc
        self.batch_grpc = batch_grpc
        self.vllm = vllm
        if self.vllm and (self.grpc or self.batch_grpc):
            sys.exit("vllm does not support grpc")
        self.device = device

        if not batch_size:
            if device == "cpu":
                batch_size = 128
            else:
                batch_size = 32  # Reduce to 8 if using 4 GPUs, 16 for 8.
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

        if 'cuda' in self.device:
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(dataset_path=self.dataset_path,
                                   total_count_override=total_sample_count)
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)
        self.load_model()

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()


    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()


    def query_api(self, input):
        headers = {
            'Content-Type': 'application/json',
        }

        json_data = {
            'model_id': self.api_model_name,
            'inputs': input,
            'parameters': {
                'max_new_tokens': 1024,
                'min_new_tokens': 1,
                'decoding_method': "GREEDY"
            },
        }

        response_code = 0
        while response_code != 200:
            try:
                response = requests.post(
                    self.api_server,
                    headers=headers,
                    json=json_data,
                    verify=False,
                )
                response_code = response.status_code
            except:
                print("connection failure")
        return json.loads(response.text)["generated_text"]
    
    def query_vllm(self, inputs):
        sampling_params = SamplingParams(
            max_tokens=128,
            use_beam_search=True,
            best_of=4,
            temperature=0,
            early_stopping=True,
        )

        outputs = self.llm.generate(inputs, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def query_api_grpc(self, input):
        resp = self.grpc_client.make_request([input], model_id=self.api_model_name)
        return resp.responses[0].text

    def query_api_batch_grpc(self, inputs):
        resps = self.grpc_client.make_request(inputs, model_id=self.api_model_name)
        return [resp.text for resp in resps.responses]

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """

        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            fname = "q" + "_".join([str(i) for i in query_ids])
            fname = f"run_outputs/{fname}.pkl"
            _p = Path(fname)
            if self.use_cached_outputs and _p.exists():
                # Read cache
                with _p.open(mode="rb") as f:
                    d = pickle.load(f)
                processed_output = d["outputs"]
                tik1 = None
                tik2 = None
                tik3 = None
                tok = None
            else:
                # Construct / collate batch
                tik1 = time.time()

                input_ids_tensor = []
                input_masks_tensor = []
                input_len = []
                for q in qitem:
                    input_ids_tensor.append(self.data_object.source_encoded_input_ids[q.index])
                    #input_masks_tensor.append(self.data_object.source_encoded_attn_masks[q.index])
                    #input_len.append(self.data_object.input_lens[q.index])
                #input_ids_tensor = torch.cat(input_ids_tensor)
                #input_masks_tensor = torch.cat(input_masks_tensor)

                #assert input_ids_tensor.shape == input_masks_tensor.shape
                #assert input_ids_tensor.shape[0] <= self.batch_size

                if self.api_server:
                    #decoded = self.tokenizer.batch_decode(input_ids_tensor)
                    #cleaned = [entry.replace('</s>','').replace('<s>','') for entry in decoded]
                    bs = len(input_ids_tensor)

                tik2 = time.time()

                if self.api_server:
                    if self.grpc:
                        if self.batch_grpc:
                            output = self.query_api_batch_grpc(input_ids_tensor)
                        else:
                            with ThreadPoolExecutor(max_workers=bs) as executor:
                                output = list(executor.map(self.query_api_grpc,input_ids_tensor))
                    elif self.vllm:
                        output = self.query_vllm(input_ids_tensor)
                    else:
                        with ThreadPoolExecutor(max_workers=bs) as executor:
                            output = list(executor.map(self.query_api,input_ids_tensor))
                else:
                    pred_output_tokens = self.model.generate(
                        input_ids=input_ids_tensor,
                        attention_mask=input_masks_tensor,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **gen_kwargs
                    )

                tik3 = time.time()

                if self.api_server:
                    processed_output = np.array(self.tokenizer(output, padding='longest')['input_ids'])
                else:
                    processed_output = self.data_object.postProcess(pred_output_tokens,
                                                                    input_seq_lens=input_len,
                                                                    query_id_list=query_ids)

            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                print(f"Samples run: {self.sample_counter}")
                if tik1:
                    print(f"\tBatchMaker time: {tik2 - tik1}")
                    print(f"\tInference time: {tik3 - tik2}")
                    print(f"\tPostprocess time: {tok - tik3}")
                    print(f"\t==== Total time: {tok - tik1}")
                else:
                    print(f"\tLoaded from cache: {_p}")


    def load_model(self):
        if self.api_server:
            if self.grpc:
                hostname = re.sub("https://|http://", "", self.api_server)
                if hostname[-1] == "/":
                    hostname = hostname[:-1]
                self.grpc_client = GrpcClient(
                    hostname,
                    443,
                    verify=False,
                )
            elif self.vllm:
                self.llm = LLM(model="/workspace/gpt-model-info/", dtype="float16")
            elif not "http" in self.api_server:
                self.api_server = "http://" + self.api_server

            if not self.api_model_name:
                sys.exit("API Server was specified but no model name was provided")
        else:
            sys.exit("ONLY API SERVER MODE SUPPORTED FOR GPT-J")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=self.amp_dtype
            )
            print("Loaded model")

            self.device = torch.device(self.device)
            if self.device == "cpu":
                self.model = self.model.to(self.device)  # Force CPU if your system has GPU and you specifically want CPU-only run

            self.model.eval()
            self.model = self.model.to(memory_format=torch.channels_last)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=True,) #changed from false

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loaded tokenizer")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl


    def predict(self,**kwargs):
        raise NotImplementedError


    def issue_queries(self, query_samples):
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
    def __init__(self, model_path=None, api_server=None, api_model_name=None, grpc=False, batch_grpc=False, dtype="bfloat16", device="cpu", total_sample_count=24576, dataset_path=None, workers=1):

        super().__init__(model_path=model_path, api_server=api_server, api_model_name=api_model_name, grpc=grpc, dtype=dtype, device=device, total_sample_count=total_sample_count, dataset_path=dataset_path, workers=workers)

        with open(f"{self.model_path}/tokenizer.json", 'r') as token_file:
            gpt_tokenizer = json.load(token_file)
        self.gpt_vocab = gpt_tokenizer["model"]["vocab"]

        self.first_token_queue = queue.Queue()
        

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

    def stream_api(self, input, response_ids):
        headers = {
            'Content-Type': 'application/json',
        }

        json_data = {
            'model_id': 'GPT-J',
            'inputs': input,
            'parameters': {
                'max_new_tokens': 1024,
                'min_new_tokens': 1,
                'decoding_method': "GREEDY"
            },
        }

        token_cache = []
        s = requests.Session()
        first = True
        with s.post(
            self.api_server,
            headers=headers,
            json=json_data,
            verify=False,
            stream=True
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode()
                    if decoded.startswith("data"):
                        token_l = json.loads(decoded[6:])["tokens"]
                        if token_l:
                            token = self.gpt_vocab[token_l[0]["text"]]
                            if first:
                                self.first_token_queue.put((token, response_ids[0]))
                                first = False
                            else:
                                token_cache.append(token)
        return token_cache

    def stream_api_grpc(self, input, response_ids):
        token_cache = []
        first = True
        resps = self.grpc_client.make_request_stream(input, model_id=self.api_model_name)
        for resp in resps:
            if resp.tokens:
                token = self.gpt_vocab[resp.tokens[0].text]
                if first:
                    self.first_token_queue.put((token, response_ids[0]))
                    first = False
                else:
                    token_cache.append(token)
        return token_cache

    def async_process_query(self, input_ids_tensor, qitem_id):
        decoded = input_ids_tensor
        response_ids = [qitem_id]
        if self.grpc:
            output_tokens = self.stream_api_grpc(decoded, response_ids)
        else:
            output_tokens = self.stream_api(decoded, response_ids)

        n_tokens = len(output_tokens)
        response_array = array.array("B", np.array(output_tokens, np.int32).tobytes())
        bi = response_array.buffer_info()
        response = [lg.QuerySampleResponse(
            qitem_id, bi[0], bi[1], n_tokens)]
        lg.QuerySamplesComplete(response)
        sys.exit()

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = self.data_object.source_encoded_input_ids[qitem.index]
            input_masks_tensor = []#self.data_object.source_encoded_attn_masks[qitem.index]

            if self.api_server:
                threading.Thread(target=self.async_process_query, args=(input_ids_tensor, qitem.id)).start()
            else:
                #TODO: This PoC is super slow with significant overhead. Best to create a patch to `generate`
                tokens_cache = []
                tokens_streamer = FirstTokenStreamer(self.first_token_queue, tokens_cache=tokens_cache, is_first_token=True, response_ids=[qitem.id])

                _ = self.model.generate(    input_ids=input_ids_tensor,
                                            attention_mask=input_masks_tensor,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            streamer = tokens_streamer,
                                            **gen_kwargs
                                            )

                output_tokens = tokens_streamer.get_out_tokens()

                n_tokens = len(output_tokens)
                response_array = array.array("B", np.array(output_tokens, np.int32).tobytes())
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(
                    qitem.id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)


    def issue_queries(self, query_samples):

        self.query_queue.put(query_samples[0])


    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()
