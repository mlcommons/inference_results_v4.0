import array
import copy
import psutil
import json
import mlperf_loadgen as lg
import threading
import time
import torch
import torch.multiprocessing as mp
import numpy as np
import sys
import os
from backend import Instance
from item import *
from utils import get_memory_usage, logger
import coco
import dataset
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler

class BaseSUT(object):
    def __init__(
        self,
        dataset_path,
        num_proc, 
        cpus_per_proc,
        total_sample_count,
        dtype="float32",
        device_type="cpu",
        scenario="Offline",
        initial_core=0,
        workers_per_proc=1,
        args=None,
        **kwargs,
    ):
        self.args = args
        self.num_proc = num_proc
        self.cpus_per_proc = cpus_per_proc
        self.initial_core = initial_core
        self.procs = [None] * self.num_proc
        self.workers_per_proc = workers_per_proc
        self.total_workers = self.num_proc * self.workers_per_proc

        self.batch_size = args.batch_size
        self.lock = mp.Lock()
        self.scenario = scenario
        self.affinity = []
        self.start_core_marker=[]
        self.cores_per_inst = []
        num_physical_cores = psutil.cpu_count(logical=False)
        num_cores_per_socket = int(num_physical_cores/2)



        if scenario=="SingleStream":
            self.affinity.append(list(range(0,num_physical_cores)))
            self.start_core_marker = [0]
            self.cores_per_inst = [num_physical_cores]
        elif scenario=="Server":

            self.affinity.append(list(range(0,num_cores_per_socket)))
            self.affinity.append(list(range(num_cores_per_socket,int(2*num_cores_per_socket))))

            self.start_core_marker = [0,num_cores_per_socket]
            self.cores_per_inst = [num_cores_per_socket,num_cores_per_socket]

        else:
            for i in range(0,num_physical_cores-self.cpus_per_proc+1,self.cpus_per_proc):
                self.affinity.append(list(range(i,i+self.cpus_per_proc)))
                self.start_core_marker.append(i)
                self.cores_per_inst.append(self.cpus_per_proc)

        self.accuracy = args.accuracy

        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.device_type = device_type

        # Model specific parameters, do not change
        self.guidance = 8
        self.steps = 20
        self.negative_prompt = "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
        self.max_length_neg_prompt = 77
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        print("args.model_path : ",args.model_path)

        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            os.path.join(args.model_path, "checkpoint_scheduler"), subfolder="scheduler"
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
                os.path.join(args.model_path, "checkpoint_pipe"),
                scheduler=self.scheduler,
                safety_checker=None,
                add_watermarker=False,
                variant="fp16" if (self.dtype == torch.float16) else None,
                torch_dtype=self.dtype,
        )

        '''self.pipe.tokenizer.padding_side = "left"
        self.pipe.tokenizer_2.padding_side = "left"'''

        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.negative_prompt_tokens = self.pipe.tokenizer(
            self.convert_prompt(self.negative_prompt, self.pipe.tokenizer),
            padding="max_length",
            max_length=self.max_length_neg_prompt,
            truncation=True,
            return_tensors="pt",
        )
        self.negative_prompt_tokens_2 = self.pipe.tokenizer_2(
            self.convert_prompt(self.negative_prompt, self.pipe.tokenizer_2),
            padding="max_length",
            max_length=self.max_length_neg_prompt,
            truncation=True,
            return_tensors="pt",
        )

        self.time_batch = sys.maxsize

        get_memory_usage("Host", "cpu")

        self.dataset = coco.Coco(
            data_path=args.dataset_path,
            name=args.dataset,
            pre_process=dataset.preprocess,
            pipe_tokenizer=self.pipe.tokenizer,
            pipe_tokenizer_2=self.pipe.tokenizer_2,
            latent_dtype=self.dtype,
            latent_device=self.device_type,
            latent_framework="torch",
            total_sample_count=total_sample_count,
            **kwargs,)
        self.warmup_dataset = coco.Coco(
            data_path=os.path.join(args.dataset_path,"warmup_dataset"),
            name=args.dataset,
            pre_process=dataset.preprocess,
            pipe_tokenizer=self.pipe.tokenizer,
            pipe_tokenizer_2=self.pipe.tokenizer_2,
            latent_dtype=self.dtype,
            latent_device=self.device_type,
            latent_framework="torch",
            total_sample_count=total_sample_count,
            **kwargs,)
            
        self.post_process = coco.PostProcessCoco()
        self.num_instances = self.num_proc
        self.insts = []
        self.input_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()
        self.init_counter = mp.Value("i", 0)
        # self.cond_var = mp.Condition(lock=mp.Lock())
        self.cv = mp.Condition(lock=self.lock)
    
    def createProcesses(self):
        """ Create 'mp' instances or processes"""

        start_core = self.initial_core
        index =0 
        for proc_idx in range(self.num_proc):
            self.procs[proc_idx] = Instance(
                model=self.pipe,
                dataset=self.dataset,
                warmup_dataset=self.warmup_dataset,
                post_process=self.post_process,
                guidance=self.guidance,
                steps=self.steps,
                negative_prompt_tokens=self.negative_prompt_tokens,
                negative_prompt_tokens_2=self.negative_prompt_tokens_2,
                input_queue=self.input_queue,
                output_queue= self.output_queue,
                lock=self.lock,
                dtype=self.dtype,
                device_type=self.device_type,
                batch_size=self.batch_size,
                core_list=[],
                affinity=self.affinity[index],
                init_counter=self.init_counter,
                proc_idx=proc_idx, 
                start_core_idx=self.start_core_marker[index], 
                cpus_per_proc=self.cores_per_inst[index], 
                workers_per_proc=self.workers_per_proc,
                cond_var=self.cv,
                enable_warmup=self.args.warmup,
                enable_profile=self.args.profile,
                scenario=self.scenario,
            )
            start_core += self.cpus_per_proc
            index+=1
    
    def startSUT(self):
        """ Creates and Starts the processes and threads"""

        # Create processes
        self.createProcesses()
        # Start processes
        logger.info("Starting processes")
        for proc in self.procs:
            proc.start()
        
        # Wait for all consumers to be ready (including if they're warming up)
        with self.cv:
            self.cv.wait_for(lambda : self.init_counter.value==self.num_proc)

        # Start Loadgen response thread
        self.response_thread = threading.Thread(target=self.response_loadgen)
        self.response_thread.daemon = True
        self.response_thread.start()        




    def convert_prompt(self, prompt, tokenizer):
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def response_loadgen(self):
        num_response = 0
        next_result = None

        while True:

            try:
                next_result = self.output_queue.get()
            except:
                print("Exception")
 

            if next_result is None:
                logger.info("Exiting response thread")
                break

            query_id = next_result.id
            result = next_result.result

            response_array_refs = []
            responses = []
            for id, out in zip(query_id, result):
                response_array = array.array("B", np.array(out, np.uint8).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                responses.append(lg.QuerySampleResponse(id, bi[0], bi[1]))
            lg.QuerySamplesComplete(responses)

    def flush_queries(self):
        pass

    def postprocess_accuracy(self, result_dict):
        self.post_process.finalize(result_dict, self.dataset)
        return result_dict

    def load_query_samples(self, sample_list):
        pass
    
    def unload_query_samples(self, sample_list):
        pass

class OfflineSUT(BaseSUT):
    def __init__(
        self,
        dataset_path,
        num_proc, 
        cpus_per_proc,
        total_sample_count,
        dtype="float32",
        device_type="cpu",
        scenario="Offline",
        initial_core=0,
        workers_per_proc=1,
        args=None,
        **kwargs,
    ):
        super().__init__(
            dataset_path,
            num_proc, 
            cpus_per_proc, 
            total_sample_count,
            dtype,
            device_type,
            scenario,
            initial_core,
            workers_per_proc,
            args,
            **kwargs,
        )

    def issue_queries(self, samples):
        i = 0
        while i < len(samples):
            
            enqueue_time = time.time()
            query_samples = samples[i : min(i + self.batch_size, len(samples))]
            query_id = [sample.id for sample in query_samples]
            query_idx = [sample.index for sample in query_samples]
            input_tokens, input_tokens_2, latents = self.dataset.get_samples(query_idx)

            self.input_queue.put(
                InputItem(
                    query_id,
                    query_idx,
                    query_samples,
                    None,
                    None,
                    input_tokens,
                    input_tokens_2,
                    latents,
                    enqueue_time,
                )
            )


            i += self.batch_size
   
    
    def stop_sut(self):
        for _ in range(self.num_instances):
            self.input_queue.put(None)

        for inst in self.insts:
            inst.join()
        # self.response_thread.join()
        self.output_queue.put(None)

class ServerSUT(BaseSUT):
    def __init__(
        self,
        dataset_path,
        num_proc, 
        cpus_per_proc,
        total_sample_count,
        dtype="float32",
        device_type="cpu",
        scenario="Offline",
        initial_core=0,
        workers_per_proc=1,
        args=None,
        **kwargs,
    ):
        super().__init__(
            dataset_path,
            num_proc, 
            cpus_per_proc, 
            total_sample_count,
            dtype,
            device_type,
            scenario,
            initial_core,
            workers_per_proc,
            args,
            **kwargs,
        )
        self.query_id=[]
        self.query_idx=[]
        self.query_sample=[]

   

    def issue_queries(self, samples):
        """ Receives queries and adds them to queue for processing"""

        enqueue_time = time.time()

        self.time_batch = min(self.time_batch,enqueue_time)


        self.query_sample.append(samples[0])
        self.query_id.append(samples[0].id)
        self.query_idx.append(samples[0].index)

        if len(self.query_id)==self.batch_size:
            item = InputItem(
                    self.query_id,
                    self.query_idx,
                    self.query_sample,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.time_batch,
                )
            self.time_batch = sys.maxsize
            self.input_queue.put(item)
            self.query_id = []
            self.query_idx = []
            self.query_sample = []
      

    def stop_sut(self):
        for _ in range(self.num_instances):
            self.input_queue.put(None)

        for inst in self.insts:
            inst.join()
        # self.response_thread.join()
        self.output_queue.put(None)