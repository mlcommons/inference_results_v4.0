import array
import collections
import logging
import multiprocessing as mp
import multiprocessing.synchronize as synchronize
import multiprocessing.sharedctypes as sharedctypes
import os
import sys
import threading
from typing import List, Optional

import mlperf_loadgen as lg
import thread_binder
import torch
from item import InputItem
from item import OutputItem
from backend import Backend, BackendServer
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT")

SERVER_IN_OFFLINE = True


class Consumer(mp.Process):
    ''' One Consumer per processor (socket) '''

    def __init__(self, model_path="", precision="int8", model_checkpoint="", dataset_path="",
                 input_queue=None, out_queue=None, lock=None, cond_var=None, init_counter=None, batch_size=None,
                 beam_size=1, proc_idx=0, start_core_idx=0, cpus_per_proc=56, cpus_per_worker=None, warmup=False,
                 total_sample_count=1000, pad_inputs=False, log_dir="", scenario="offline", logical_cores_start=-1):
        mp.Process.__init__(self)
        assert cpus_per_worker is not None
        self.cpus_per_worker: List[int] = cpus_per_worker

        self.num_workers = len(cpus_per_worker)
        self.model_path = model_path
        self.task_queue = input_queue
        self.out_queue = out_queue
        self.lock = lock
        assert init_counter is not None
        self.init_counter: sharedctypes.Synchronized[int] = init_counter
        assert batch_size is not None
        self.batch_size: List[int] = batch_size
        self.beam_size = beam_size
        self.proc_idx = proc_idx
        self.num_cores = cpus_per_proc
        assert cpus_per_proc == sum(cpus_per_worker)
        self.start_core_idx = start_core_idx
        self.end_core_idx = start_core_idx + self.num_cores - 1
        self.affinity = list(range(start_core_idx, start_core_idx + cpus_per_proc))
        self.logical_cores_start = logical_cores_start
        if logical_cores_start >= 0:
            self.affinity.extend(list(map(lambda x: x + logical_cores_start, self.affinity)))

        self.dataset_path = dataset_path

        self.model: List[Optional[Backend]] = [None for _ in range(self.num_workers)]
        self.workers = []
        self.out_queue = out_queue
        self.warmup = warmup
        self.latencies = collections.defaultdict(list)

        self.total_sample_count = total_sample_count
        self.pad_inputs = pad_inputs
        self.log_dir = log_dir
        self.scenario = scenario

        self.precision = precision
        self.model_checkpoint = model_checkpoint
        assert cond_var is not None
        self.cond_var: synchronize.Condition = cond_var

    def doWarmup(self):
        warmup_data = self.data_obj.getWarmupSamples()
        log.info("Starting warmup")

        for model in self.model:
            assert model is not None and self.scenario == "offline"
            for i, (input_ids, input_len, attention_mask) in enumerate(warmup_data):
                log.info("{} iter done".format(i))
                _ = model.predict(input_ids, attention_mask)

        log.info("Process {} Warmup Completed".format(self.proc_idx))
        with self.cond_var:
            self.init_counter.value += 1
            self.cond_var.notify()

    def handleTasks(self, i: int, task_queue: mp.JoinableQueue, result_queue, pid, start_core: int, num_cores: int, batch_size: int):
        """ Woker's "main function" """
        log_file = os.path.join(self.log_dir, f"log-worker-{pid}-{i}.log")
        log.info("Worker logging to %s", log_file)
        log_fh = open(log_file, 'w', encoding='utf-8')
        os.dup2(log_fh.fileno(), sys.stdout.fileno())
        os.dup2(log_fh.fileno(), sys.stderr.fileno())
        cores_affinity = list(range(start_core, start_core + num_cores))
        backend_n_threads = num_cores
        if self.logical_cores_start >= 0:
            cores_affinity.extend(list(map(lambda x: x + self.logical_cores_start, cores_affinity)))
            backend_n_threads *= 2
        thread_binder.set_worker_affinity(cores_affinity)

        model_i = None
        if self.scenario == "server" or SERVER_IN_OFFLINE:
            model_i = BackendServer(
                self.model_path,
                batch_size,
                self.beam_size,
                proc_idx=self.proc_idx,
                cores_num=backend_n_threads
            )
        else:
            model_i = Backend(
                self.model_path,
                batch_size,
                self.beam_size,
                proc_idx=self.proc_idx,
                cores_num=backend_n_threads
            )
        self.model[i] = model_i
        log.info(
            f"model_path: {self.model_path}\n"
            f"batch_size: {batch_size}\n"
            f"beam_size: {self.beam_size}\n"
            f"proc_idx: {self.proc_idx}\n"
            f"start_core_idx: {start_core}\n"
            f"cores_affinity: {cores_affinity}\n"
        )

        # Load model
        if self.scenario == "server" or SERVER_IN_OFFLINE:
            ws: sharedctypes.Synchronized[int] = mp.Value("i", 0)
            ws_lock = mp.Lock()
            wsc = mp.Condition(lock=ws_lock)

            def server_response(res, working_size):
                q_ids = [r.id for r in res]
                q_outs = [np.array(r.token_ids) for r in res]
                result = OutputItem(q_ids, q_outs)
                result_queue.put(result)
                task_queue.task_done()
                # qqouts = [r.token_ids for r in res]
                # print("back q_ids: {}, back results: {}".format(q_ids, qqouts), flush=True)
                log.info("working_size %d", working_size)
                with wsc:
                    ws.value = working_size
                    wsc.notify()

            log.info("Starting server")
            model_i.loadServer(server_response, lambda: thread_binder.set_worker_affinity(cores_affinity))
            log.info("Server started")
        else:
            log.info("Loading model")
            model_i.loadModel()
            log.info("Model loaded")

        # Do Warmup
        if False:  # self.warmup:
            self.doWarmup()
        else:
            with self.cond_var:
                self.init_counter.value += 1
                self.cond_var.notify()

        while True:
            try:
                if self.scenario == "server" or SERVER_IN_OFFLINE:
                    max_req = batch_size if self.scenario == "server" else batch_size + 1
                    with wsc:
                        wsc.wait_for(lambda: ws.value < max_req)

                next_task: InputItem = task_queue.get()
                if next_task is None:
                    if self.scenario == "server" or SERVER_IN_OFFLINE:
                        log.info("server next task none...")
                        while not model_i.Empty():
                            time.sleep(1)
                    log.info("Exiting worker thread: %d", i)
                    break

                query_id_list = next_task.query_id_list
                sample_index_list = next_task.sample_index_list
                input_seq_lens = next_task.input_seq_lens

                input_ids, input_seq_lens, attention_mask = self.data_obj.getSamples(sample_index_list)
                if self.scenario == "server" or SERVER_IN_OFFLINE:
                    print(f"query_id_list: {query_id_list}, input_ids: {input_ids}", flush=True)
                    model_i.add_query(input_ids, attention_mask=attention_mask, query_id_list=query_id_list)
                    with wsc:
                        ws.value += 1
                        wsc.notify()
                else:
                    output = model_i.predict(input_ids, attention_mask=attention_mask)

                    result = OutputItem(query_id_list, [np.array(o) for o in output])

                    # import sys
                    # import pdb
                    # class ForkedPdb(pdb.Pdb):
                    #     """A Pdb subclass that may be used
                    #     from a forked multiprocessing child

                    #     """
                    #     def interaction(self, *args, **kwargs):
                    #         _stdin = sys.stdin
                    #         try:
                    #             sys.stdin = open('/dev/stdin')
                    #             pdb.Pdb.interaction(self, *args, **kwargs)
                    #         finally:
                    #             sys.stdin = _stdin
                    # ForkedPdb().set_trace()
                    result_queue.put(result)
                    task_queue.task_done()
            except Exception as e:
                import traceback
                log.error(e)
                print(traceback.format_exc())
                task_queue.task_done()
                break
        sys.stdout.close()

    def run(self):
        os.sched_setaffinity(0, self.affinity)
        from dataset import Dataset  # Note: every socket needs its own dataset
        self.data_obj = Dataset(self.dataset_path, model_checkpoint_path=self.model_checkpoint,
                                total_sample_count=self.total_sample_count, pad_inputs=self.pad_inputs)

        # Load Dataset
        log.info("Loading Dataset")
        self.data_obj.loadDataset()

        start_core = self.start_core_idx

        assert sum(self.cpus_per_worker) == self.num_cores
        for i in range(self.num_workers):
            log.info("Creating worker %d", i)
            worker_cores = self.cpus_per_worker[i]

            worker = mp.Process(target=self.handleTasks, args=(i, self.task_queue,
                                self.out_queue, self.pid, start_core, worker_cores, self.batch_size[i]))

            log.info("Created worker %d", i)
            self.workers.append(worker)
            start_core += worker_cores

        for w in self.workers:
            w.start()

        for w in self.workers:
            w.join()  # wait for every work's end (get a None from the queue)

        log.info("Exiting consumer process %d", os.getpid())


class SUT(object):
    def __init__(self, num_proc, cpus_per_worker: List[int], model_path="", initial_core=0, batch_size: List[int] = None, beam_size=1,
                 dataset_path="", workers_per_proc=1, warmup=False, precision="int8", model_checkpoint="",
                 total_sample_count=1000, pad_inputs=False, log_dir="", scenario="offline", logical_cores_start=-1):
        self.model_path = model_path
        self.num_proc = num_proc
        assert len(cpus_per_worker) == workers_per_proc
        self.cpus_per_proc = sum(cpus_per_worker)
        self.cpus_per_worker = cpus_per_worker
        self.initial_core = initial_core
        self.procs: List[Optional[Consumer]] = [None] * self.num_proc
        self.workers_per_proc = workers_per_proc
        self.warmup = warmup
        self.total_workers = self.num_proc * self.workers_per_proc

        self.precision = precision
        self.model_checkpoint = model_checkpoint

        assert batch_size is not None
        assert len(batch_size) == workers_per_proc
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.dataset_path = dataset_path

        self.total_sample_count = total_sample_count
        self.pad_inputs = pad_inputs
        self.log_dir = log_dir
        self.scenario = scenario
        self.logical_cores_start = logical_cores_start

        self.lock = mp.Lock()
        self.init_counter: sharedctypes.Synchronized[int] = mp.Value("i", 0)  # type: ignore
        self.input_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()

        self.cv = mp.Condition(lock=self.lock)

    def flushQueries(self):
        """Called immediately after the last call to IssueQuery in a series is made.

        This doesn't necessarily signify the end of the test since there may be multiple series involved during a test;
        for example in accuracy mode. Clients can use this to flush any deferred queries immediately, rather than
        waiting for some timeout. This is especially useful in the server scenario.
        """
        pass

    # def processLatencies(self, latencies):
    #     pass

    def loadSamplesToRam(self, query_samples):
        '''Loads the requested query samples into memory.

        Paired with calls to UnloadSamplesFromRam.
        In the MultiStream scenarios:
          * Samples will appear more than once.
          * SystemUnderTest::IssueQuery will only be called with a set of samples
            that are neighbors in the vector of samples here, which helps
            SUTs that need the queries to be contiguous.
        In all other scenarios:
          * A previously loaded sample will not be loaded again.
        '''
        pass

    def unloadSamplesFromRam(self, query_samples):
        '''Unloads the requested query samples from memory.

        In the MultiStream scenarios:
            * Samples may be unloaded the same number of times they were loaded; however, if the implementation de-dups
            loaded samples rather than loading samples into contiguous memory, it may unload a sample the first time
            they see it unloaded without a refcounting scheme, ignoring subsequent unloads. A refcounting scheme would
            also work, but is not a requirement.
        In all other scenarios:
            * A previously unloaded sample will not be unloaded again.
        '''
        pass

    def stopSUT(self):
        """ Stops processes and threads and exit """

        log.info("Stopping all %d workers...", self.total_workers)
        for _ in range(self.total_workers):
            self.input_queue.put(None)

        for proc in self.procs:
            assert proc is not None
            proc.join()

        self.output_queue.put(None)

    def startSUT(self):
        """ Creates and Starts the processes and threads"""

        # Create processes
        log.info("Creating processes")
        self.createProcesses()

        # Start processes
        log.info("Starting processes")
        for proc in self.procs:
            assert proc is not None
            proc.start()

        # Wait for all workers to be ready (including if they're warming up)
        with self.cv:
            self.cv.wait_for(lambda: self.init_counter.value == self.num_proc * self.workers_per_proc)

        # Start Loadgen response thread
        self.response_thread = threading.Thread(target=self.responseLoadgen)
        self.response_thread.start()

    def responseLoadgen(self):
        while True:
            next_task = self.output_queue.get()

            if next_task is None:
                log.info('Exiting response thread')
                break

            query_id_list = next_task.query_id_list
            processed_result = next_task.result
            array_type_code = next_task.array_type_code

            for id, out in zip(query_id_list, processed_result):
                response_array = array.array(array_type_code, out.tobytes())
                bi = response_array.buffer_info()
                responses = [lg.QuerySampleResponse(id, bi[0], bi[1]*response_array.itemsize)]
                lg.QuerySamplesComplete(responses)

    def createProcesses(self):
        """ Create 'mp' instances or processes"""

        start_core = self.initial_core
        for proc_idx in range(self.num_proc):
            self.procs[proc_idx] = Consumer(
                self.model_path, self.precision, self.model_checkpoint, self.dataset_path,
                self.input_queue, self.output_queue, self.lock, self.cv, self.init_counter, self.batch_size,
                self.beam_size, proc_idx, start_core, self.cpus_per_proc, self.cpus_per_worker, warmup=self.warmup,
                total_sample_count=self.total_sample_count, pad_inputs=self.pad_inputs, log_dir=self.log_dir,
                scenario=self.scenario, logical_cores_start=self.logical_cores_start)

            start_core += self.cpus_per_proc

    def issueQueries(self, query_samples):
        """Lets the loadgen issue N samples to the SUT.

        The SUT may either a) return immediately and signal completion at a later time on another thread or b) it may
        block and signal completion on the current stack. The load generator will handle both cases properly.
        Note: The data for neighboring samples may or may not be contiguous depending on the scenario.

        typedef uintptr_t ResponseId;
        typedef size_t QuerySampleIndex;
        struct QuerySample {
            ResponseId id;
            QuerySampleIndex index;
        };
        """
        query_batch = 1 if SERVER_IN_OFFLINE else self.batch_size[0]

        # add queries to SUT's queue as batches
        for i in range(0, len(query_samples), query_batch):  # todo: use gcd
            self.input_queue.put(InputItem(
                [s.id for s in query_samples[i:i+query_batch]],
                [s.index for s in query_samples[i:i+query_batch]]
            ))
        log.info("input_queue +%d (%d)", len(query_samples), self.input_queue.qsize())
