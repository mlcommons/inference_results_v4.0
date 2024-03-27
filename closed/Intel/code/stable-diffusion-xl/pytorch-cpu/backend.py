import os
import psutil
import time
import torch
import torch.multiprocessing as mp
import time
from contextlib import nullcontext
from item import *
from utils import get_memory_usage, profile_handler, logger
from typing import Optional, List, Union
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
import numpy as np
import array
import thread_binder
from numa import memory

class Instance(mp.Process):
    def __init__(
        self,
        model=None,
        dataset=None,
        warmup_dataset=None,
        post_process=None,
        guidance=8,
        steps=20,
        negative_prompt_tokens=None,
        negative_prompt_tokens_2=None,
        input_queue=None,
        output_queue=None,
        lock=None,
        dtype=torch.float32,
        device_type="cpu",
        batch_size=1,
        core_list=[],
        affinity=[],
        init_counter=1,
        proc_idx=None, 
        start_core_idx=0, 
        cpus_per_proc=60, 
        workers_per_proc=1,
        cond_var=None,
        enable_warmup=False,
        enable_profile=False,
        scenario="Offline",
        accuracy=False,
    ):
        mp.Process.__init__(self)
        self.num_workers = workers_per_proc
        self.lock = lock
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        self.num_cores = cpus_per_proc
        self.start_core_idx = start_core_idx
        self.end_core_idx = start_core_idx + self.num_cores - 1
        # self.affinity = list(range(self.start_core_idx, self.start_core_idx + self.num_cores))
        self.affinity=affinity
        self.cpus_per_worker = self.num_cores // self.num_workers
        self.workers = []
        self.numa_node_dict = {}

        self.accuracy = accuracy
        self.scenario = scenario
        self.batch_size = batch_size
        self.model = model
        self.dataset = dataset
        self.warmup_dataset = warmup_dataset
        self.post_process = post_process
        self.guidance = guidance
        self.steps = steps
        self.negative_prompt_tokens = negative_prompt_tokens
        self.negative_prompt_tokens_2 = negative_prompt_tokens_2
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.dtype = dtype
        self.enable_amp = self.dtype == torch.float16
        self.device_type = device_type

        self.device = torch.device(self.device_type)
        self.cond_var = cond_var
        self.enable_warmup = enable_warmup
        self.enable_profile = enable_profile
        self.profile_iter = 0
        self.profile_ctx = nullcontext()
        self.reach_end = False


    def init_model(self):
  
        import intel_extension_for_pytorch as ipex

        logger.info(f"Casting model to {self.device}")
        self.model = self.model.to(self.device, self.dtype)
        self.dataset.load_latent(self.device)
        self.model.set_progress_bar_config(disable=True)
        get_memory_usage(f"{self.device}", self.device_type, self.device)

        logger.info(f"Optimizing model on {self.device}")


        self.model.text_encoder = ipex.optimize(self.model.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
        self.model.unet = ipex.optimize(self.model.unet.eval(), dtype=torch.bfloat16, weights_prepack=False)
        self.model.vae = ipex.optimize(self.model.vae.eval(), dtype=torch.bfloat16, inplace=True)
        ipex._set_compiler_backend("torchscript")
        # self.model.text_encoder = torch.compile(self.model.text_encoder,backend="ipex")
        self.model.unet = torch.compile(self.model.unet, backend="ipex")
        # self.model.vae = torch.compile(self.model.vae, backend="ipex")


        get_memory_usage(f"{self.device}", self.device_type, self.device)


    def do_warmup(self): # TODO: Fix the samples used??
 

        for i in range(5):
            input_tokens, input_tokens_2, latents = self.warmup_dataset.get_samples([i for i in range(self.batch_size)])
            _ = self.predict(input_tokens, input_tokens_2, latents)
            
        #self.dataset.unload_query_samples(None)
        
        #latency = time.time() - start

        logger.info("Process {} Warmup Completed".format(self.pid))

        with self.cond_var:
            self.init_counter.value += 1
            self.cond_var.notify()
     

    def encode_tokens(
        self,
        pipe: StableDiffusionXLPipeline,
        text_input: torch.Tensor,
        text_input_2: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[torch.Tensor] = None,
        negative_prompt_2: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        device = device or pipe._execution_device
        batch_size = text_input.shape[0]

        # Define tokenizers and text encoders
        tokenizers = (
            [pipe.tokenizer, pipe.tokenizer_2]
            if pipe.tokenizer is not None
            else [pipe.tokenizer_2]
        )
        text_encoders = (
            [pipe.text_encoder, pipe.text_encoder_2]
            if pipe.text_encoder is not None
            else [pipe.text_encoder_2]
        )

        if prompt_embeds is None:
            #text_input_2 = text_input_2 or text_input

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            text_inputs_list = [text_input, text_input_2]
            for text_inputs, tokenizer, text_encoder in zip(
                text_inputs_list, tokenizers, text_encoders
            ):
                text_input_ids = text_inputs#.input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(device), output_hidden_states=True
                )
                
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = (
            negative_prompt is None and pipe.config.force_zeros_for_empty_prompt
        )
        if (
            do_classifier_free_guidance
            and negative_prompt_embeds is None
            and zero_out_negative_prompt
        ):
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt_inputs = (
                negative_prompt.input_ids.repeat(batch_size, 1)
                #if (len(negative_prompt.input_ids.shape) == 1)
                #else negative_prompt.input_ids
            )
            negative_prompt_2_inputs = (
                negative_prompt_2.input_ids.repeat(batch_size, 1)
                #if (len(negative_prompt_2.input_ids.shape) == 1)
                #else negative_prompt_2.input_ids
            )

            uncond_inputs = [negative_prompt_inputs, negative_prompt_2_inputs]

            negative_prompt_embeds_list = []
            for uncond_input, tokenizer, text_encoder in zip(
                uncond_inputs, tokenizers, text_encoders
            ):
                negative_prompt_embeds = text_encoder(
                    uncond_input.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if pipe.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(
                dtype=pipe.text_encoder_2.dtype, device=device
            )
        else:
            prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if pipe.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=pipe.text_encoder_2.dtype, device=device
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=pipe.unet.dtype, device=device
                )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(bs_embed * num_images_per_prompt, -1)
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def predict(self, input_tokens, input_tokens_2, latents):
        images = []
        # print("length of input_tokens : ",len(input_tokens))
        with torch.no_grad(): 
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_tokens(
                self.model,
                input_tokens,
                input_tokens_2,
                negative_prompt=self.negative_prompt_tokens,
                negative_prompt_2=self.negative_prompt_tokens_2,
            )
            # print("length of prompt_embeds : ",len(prompt_embeds))
            tic = time.time()
            images = self.model(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
                output_type="pt",
                latents=latents,
            ).images
            toc = time.time()
            # print("Time taken : ",toc-tic)

        return images


    def handleTasks(self, i, task_queue, result_queue, pid, start_core, num_cores,affinity):

        # thread_binder.bind_thread(start_core, num_cores)
        thread_binder.bind_thread(len(affinity),affinity)
        worker_name = str(pid) + "-" + str(i)

        # Do Warmup
        if self.enable_warmup:
            self.do_warmup()

        else:
            with self.cond_var:
                self.init_counter.value += 1
                self.cond_var.notify()

        tw=0
        t1=0
        t0=0
        ts=0
        # tw = time.time()
        while True:
            try:
                tw = time.time()
                next_task = task_queue.get()

                if next_task is None:
                    logger.info("Exiting worker thread : {}".format(i))
                    break
                t0 = time.time()
                t_receipt = next_task.receipt_time
                query_id = next_task.id_list
                query_idx = next_task.idx_list
                input_tokens = next_task.input_tokens
                input_tokens_2 = next_task.input_tokens_2
                latents = next_task.latents

                input_tokens, input_tokens_2, latents = self.dataset.get_samples(query_idx)
                # print("Before prediction inside handleTask")
                t1 = time.time()
                outputs = self.predict(input_tokens, input_tokens_2, latents)

                t2 = time.time()
                results = [
            (t.cpu().permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
            for t in outputs
        ]
                result_queue.put(OutputItem(query_id, results))
                t_output_queue = time.time()

                # logger.info(f"query_idx: {query_idx} || Instance ID : {pid} || Numa Node beginning : {self.numa_node_dict[pid]} || receipt-finish total time : {t_output_queue-t_receipt:6.2f} || Actual predict latency : {t2-t1:6.2f} || fetch from input queue wait: {t0-t_receipt:6.2f} ")

                # print(f"query_idx: {query_idx} ||  Numa Nodes : {self.numa_node_dict[pid]}-{self.numa_node_dict[pid]+self.num_cores} || receipt-finish total time : {t_output_queue-t_receipt:6.2f} || Actual predict latency : {t2-t1:6.2f} fetch from input queue wait: {t0-t_receipt:6.2f} ")

                task_queue.task_done()

            except Exception as ex:
                # Error occured
                logger.error(ex)
                break
                self.terminate()
                sys.exit(1)



    def run(self):

        # self.proc_idx = self.pid
        # self.pid = os.getpid()
        num_numa_nodes = len(memory.get_membind_nodes())
        num_physical_cores = psutil.cpu_count(logical=False)
        num_cores_per_socket = int(num_physical_cores/2)
        num_cores_per_node = int(num_cores_per_socket/2)

        print("self.affinity : ",self.affinity)
        os.sched_setaffinity(os.getpid(), self.affinity)
        

        if num_numa_nodes==2:
            # 2 NUMA Nodes (SNC disabled)

            if self.scenario=="SingleStream":
                memory.set_membind_nodes(0,1)
            elif self.scenario=="Server":
                if self.affinity[0]<=num_cores_per_socket-1:
                    memory.set_membind_nodes(0)
                else:
                    memory.set_membind_nodes(1)
            else:
                if self.affinity[0]<num_cores_per_socket:
                    memory.set_membind_nodes(0)
                else:
                    memory.set_membind_nodes(1)

        elif num_numa_nodes==4:
            # 4 NUMA Nodes (SNC enabled)

            if self.scenario=="SingleStream":
                memory.set_membind_nodes(0,1,2,3)
            elif self.scenario=="Server":

                if self.affinity[0]==0:
                    memory.set_membind_nodes(0,1)
                else:
                    memory.set_membind_nodes(2,3)

            else:
                if self.affinity[-1]<num_cores_per_node:
                    memory.set_membind_nodes(0)
                elif self.affinity[-1]>=num_cores_per_node and self.affinity[-1]<num_cores_per_socket:
                    memory.set_membind_nodes(1)
                elif self.affinity[-1]>=num_cores_per_socket and self.affinity[-1]<num_cores_per_node+num_cores_per_socket:
                    memory.set_membind_nodes(2)
                else:
                    memory.set_membind_nodes(3)
            

        self.numa_node_dict[os.getpid()] = self.affinity[0]


        self.init_model()

        start_core = self.start_core_idx
        cores_left = self.num_cores
        cores_rem = self.num_cores - self.num_workers * self.cpus_per_worker


        print("num_workers : ",self.num_workers)
        for i in range(self.num_workers):
                logger.info("Creating worker {}".format(i))
             
                worker_cores = self.cpus_per_worker + max(0, min(1, cores_rem))
                cores_left -= self.cpus_per_worker
                
                worker = mp.Process(target=self.handleTasks, args=(i, self.input_queue, self.output_queue, os.getpid(), start_core, worker_cores,self.affinity))

                self.workers.append(worker)
                start_core += self.cpus_per_worker
                cores_rem -= 1

        for w in self.workers:
            w.start()

        for w in self.workers:
            w.join()

        logger.info("{} : Exiting consumer process".format(os.getpid()))     

       
