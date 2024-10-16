import logging
import os
import time
from typing import List

import numpy as np
from neural_speed import Model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BACKEND")

NEURAL_SPEED_VERBOSE = int(os.environ.get("NEURAL_SPEED_VERBOSE", "-1") or "-1")
os.environ["NEURAL_SPEED_VERBOSE"] = str(NEURAL_SPEED_VERBOSE)


class Backend(object):
    SEED = 1234
    N_PREDICT = 128
    N_CTX = 2048
    TOP_K = 40
    TOP_P = 1.
    TEMPERATURE = .9
    REPETITION_PENALTY = 1.5
    MIN_NEW_TOKENS = 30
    LENGTH_PENALTY = 1.
    DO_EARLY_STOPPING = True

    def __init__(self, model_path, batch_size, beam_size, proc_idx=0, cores_num=56):
        self.model = Model()
        self.model.model_type = 'gptj'
        self.model_path = model_path
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.proc_idx = proc_idx
        self.cores_num = cores_num
        log.info("cores num: {}".format(self.cores_num))
        np.set_printoptions(threshold=10)

    def loadModel(self):
        '''
        void init_model(const std::string& model_path, int n_predict, int n_batch, int ctx_size, int seed, int threads,
                        float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p, float temperature,
                        int min_new_tokens, float length_penalty, bool early_stopping, int n_keep, int n_discard,
                        bool shift_roped_k, int batch_size, model_vocab::id pad_token, const std::string& memory_dtype,
                        const bool& continuous_batching, const int& max_request_num);
        '''

        log.info("\nmodel path: " + self.model_path + "\n")
        log.info("\ncores_num: " + str(self.cores_num) + "\n")
        model_path = f'{self.model_path}{self.proc_idx}'.encode('utf-8')
        self.model.init_from_bin(
            self.model.model_type,
            model_path,
            max_new_tokens=self.N_PREDICT,
            n_batch=self.N_CTX,
            ctx_size=self.N_CTX,
            seed=self.SEED,
            threads=self.cores_num,
            repetition_penalty=self.REPETITION_PENALTY,
            num_beams=self.beam_size,
            do_sample=False,
            top_k=self.TOP_K,
            top_p=self.TOP_P,
            temperature=self.TEMPERATURE,
            min_new_tokens=self.MIN_NEW_TOKENS,
            length_penalty=self.LENGTH_PENALTY,
            early_stopping=self.DO_EARLY_STOPPING,
            n_keep=0,
            n_discard=-1,
            shift_roped_k=False,
            batch_size=self.batch_size,
            pad_token=-1,
            memory_dtype="auto",
            continuous_batching=True,
            max_request_num=self.batch_size,
            model_scratch_enlarge_scale=max(1, self.batch_size / 6.)  # double when batchsize=12 for safety
        )

    def predict(self, input_batch: List[List[float]], attention_mask=None):
        """ Runs inference on 'input_batch'

        std::vector<std::vector<model_token>> generate(const std::vector<std::vector<model_token>>& input_ids);
        """
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('/home/wangzhe/dingyi/models/finetuned-gptj', trust_remote_code=True, padding_side="left")
        # tokenizer.decode(input_ids[0])
        # tokenizer.decode(output[0])
        start = 0
        if NEURAL_SPEED_VERBOSE >= 0:
            lengths = [len(xs) for xs in input_batch]
            log.info("====================\n"
                     f"predict ({sum(lengths)}: {lengths}): {[np.array(xs) for xs in input_batch]}")
            start = time.time()
        ret = self.model.model.generate(input_batch)
        self.model.model.reinit()
        if NEURAL_SPEED_VERBOSE >= 0:
            if NEURAL_SPEED_VERBOSE in [0, 1]:
                self.model.model.print_time()
                self.model.model.reset_time()
            lengths = [len(xs) for xs in ret]
            log.info("====================\n"
                     f"response ({sum(lengths)}: {lengths}) ({time.time() - start :.4f} s): {[np.array(xs) for xs in ret]}")
        return ret


class BackendServer(object):
    SEED = 1234
    N_PREDICT = 128
    N_CTX = 2048
    TOP_K = 40
    TOP_P = 1.
    TEMPERATURE = .9
    REPETITION_PENALTY = 1.5
    MIN_NEW_TOKENS = 30
    LENGTH_PENALTY = 1.
    DO_EARLY_STOPPING = True

    def __init__(self, model_path, batch_size, beam_size, proc_idx=0, cores_num=56):
        self.server = None
        self.model_path = model_path
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.proc_idx = proc_idx
        self.cores_num = cores_num
        log.info("cores num: {}".format(self.cores_num))
        np.set_printoptions(threshold=10)
        import neural_speed.gptj_cpp as cpp
        self.cpp = cpp

    def loadServer(self, response_func, init_cb):
        '''
        void init_model(const std::string& model_path, int n_predict, int n_batch, int ctx_size, int seed, int threads,
                        float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p, float temperature,
                        int min_new_tokens, float length_penalty, bool early_stopping, int n_keep, int n_discard,
                        bool shift_roped_k, int batch_size, model_vocab::id pad_token, const std::string& memory_dtype,
                        const bool& continuous_batching, const int& max_request_num);
        '''

        log.info("\nmodel path: " + self.model_path + "\n")
        log.info("\ncores_num: " + str(self.cores_num) + "\n")
        model_path = f'{self.model_path}{self.proc_idx}'.encode('utf-8')
        self.server = self.cpp.ModelServer(
            response_func,
            model_path,
            max_new_tokens=self.N_PREDICT,
            n_batch=self.N_CTX,
            ctx_size=self.N_CTX,
            seed=self.SEED,
            threads=self.cores_num,
            repetition_penalty=self.REPETITION_PENALTY,
            num_beams=self.beam_size,
            do_sample=False,
            top_k=self.TOP_K,
            top_p=self.TOP_P,
            temperature=self.TEMPERATURE,
            min_new_tokens=self.MIN_NEW_TOKENS,
            length_penalty=self.LENGTH_PENALTY,
            early_stopping=self.DO_EARLY_STOPPING,
            n_keep=0,
            n_discard=-1,
            shift_roped_k=False,
            batch_size=self.batch_size,
            pad_token=-1,
            memory_dtype="auto",
            continuous_batching=True,
            max_request_num=self.batch_size,
            model_scratch_enlarge_scale=max(1, self.batch_size / 6.),  # double when batchsize=12 for safety
            init_cb=init_cb
        )

    def add_query(self, input_batch: List[List[float]], attention_mask=None, query_id_list=None):
        """ Runs inference on 'input_batch'

        std::vector<std::vector<model_token>> generate(const std::vector<std::vector<model_token>>& input_ids);
        """
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('/home/wangzhe/dingyi/models/finetuned-gptj', trust_remote_code=True, padding_side="left")
        # tokenizer.decode(input_ids[0])
        # tokenizer.decode(output[0])
        if NEURAL_SPEED_VERBOSE >= 0:
            lengths = [len(xs) for xs in input_batch]
            log.info("====================\n"
                     f"predict ({sum(lengths)}: {lengths}): {[np.array(xs) for xs in input_batch]}")
        ids_list = query_id_list if query_id_list else [i for i in range(len(input_batch))]
        queries = [self.cpp.Query(ids_list[i], input_batch[i]) for i in range(len(input_batch))]
        self.server.issueQuery(queries)

    def Empty(self):
        assert self.server is not None
        return self.server.Empty()
