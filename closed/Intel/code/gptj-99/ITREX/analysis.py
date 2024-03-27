"""Analysis script for profiling summarization"""
import os
import os.path
import re
import dataclasses
from typing import Dict, List
import tqdm
import pandas as pd


LOG_DIR = os.environ.get(
    'LOG_DIR', './offline-accuracy-output-JF5300-B11A335T-batch-4-procs-2-ins-per-proc-7-01-12-06-40')
print(LOG_DIR)


PATTERN_FILENAME = re.compile(r'^log-worker-\d+-\d+.log$')
PATTERN_REQUEST = re.compile(r'INFO:BACKEND:=+\npredict \(.+\n(?:.*\n)+?INFO:BACKEND:=+\nresponse \(.+\n')
PATTERN_PREDICTION = re.compile(r'^prediction\s*(\d+),\s*time:\s*((?:\d|\.)+)\s*ms\s*$', re.MULTILINE)
PATTERN_REQ_STAT = re.compile(r'^predict\s+\((\d+):\s+\[(.+?)\]\)', re.MULTILINE)
PATTERN_RES_STAT = re.compile(r'^response \((\d+):\s+\[(.+?)\]\) \(((?:\d|\.)+) s\)', re.MULTILINE)
PATTERN_LIST_SPLITTER = re.compile(r'\s*,\s*')
PATTERN_MODEL_TIME_MATCH = re.compile(r'^model_print_timings:\s+(.+ time) =\s+((?:\d|\.)+)\s+?ms', re.MULTILINE)
PATTERN_MODEL_TIME_DEL = re.compile(r'^model_print_timings:.+', re.MULTILINE)
PATTERN_PERF_SPLITTER = re.compile(r'^=+$', re.MULTILINE)
PATTERN_PERF_ENTRY = re.compile(r'perf_total_per_op_us\[\s*(.+)\s*\]\s*=\s*((?:\d|\.)+)\s*ms', re.MULTILINE)


@dataclasses.dataclass
class RequestStat:
    prediction_times: Dict[int, float]
    req_len: int
    req_lens: List[int]
    res_len: int
    res_lens: List[int]
    res_time: float
    model_times: Dict[str, float]
    op_perf: List[Dict[str, float]]

    def __init__(self, req_log_str: str):
        self.prediction_times = {int(kv[0]): float(kv[1]) for kv in re.findall(PATTERN_PREDICTION, req_log_str)}
        (req_len, req_lens), = re.findall(PATTERN_REQ_STAT, req_log_str)
        self.req_len = int(req_len)
        self.req_lens = [int(i) for i in re.split(PATTERN_LIST_SPLITTER, req_lens)]
        (res_len, res_lens, res_time),  = re.findall(PATTERN_RES_STAT, req_log_str)
        self.res_len = int(res_len)
        self.res_lens = [int(i) for i in re.split(PATTERN_LIST_SPLITTER, res_lens)]
        self.res_time = float(res_time)
        self.model_times = {kv[0]: float(kv[1]) for kv in re.findall(PATTERN_MODEL_TIME_MATCH, req_log_str)}

        req_log_str_pure = re.sub(PATTERN_MODEL_TIME_DEL, '', req_log_str).replace('\n\n\n\n\n\n', '')
        self.op_perf = [{kv[0]: float(kv[1]) for kv in re.findall(PATTERN_PERF_ENTRY, step_str)}
                        for step_str in re.split(PATTERN_PERF_SPLITTER, req_log_str_pure)]
        self.op_perf = [i for i in self.op_perf if len(i)]


_, _, files = next(os.walk(LOG_DIR))
worker_logs = [f for f in files if re.match(PATTERN_FILENAME, f) is not None]


stats = []
for worker_log in tqdm.tqdm(worker_logs):
    with open(os.path.join(LOG_DIR, worker_log), encoding='utf-8') as fh:
        log_str = fh.read()
    req_strs = re.findall(PATTERN_REQUEST, log_str)
    # print(f'{len(req_strs)} requests in {worker_log}')
    for req_str in req_strs:
        stats.append(RequestStat(req_str))

total_backend_time = sum(x.res_time for x in stats)
inst_num = len(worker_logs)
equivalent_time = total_backend_time / inst_num
total_sample = sum(len(x.req_lens) for x in stats)
samples_per_second = total_sample / equivalent_time
print('total_backend_time:', total_backend_time)
print('inst_num:', inst_num)
print('equivalent_time:', equivalent_time)
print('total_sample:', total_sample)
print('samples_per_second:', samples_per_second)

df = pd.DataFrame(q.op_perf for q in stats).rename_axis('req_id').rename_axis('step', axis="columns")
df = df.stack().rename('perf').reset_index()
df = df.join(pd.json_normalize(df['perf'])).drop('perf', axis='columns')
df = df.set_index(['req_id', 'step'])


df_total = df.groupby('step').agg('sum')

total_token_time = df_total.loc[:].sum(axis=0)
next_token_time = df_total.loc[1:].sum(axis=0)
next_token_1to16 = df_total.loc[1:16].sum(axis=0)
next_token_1to32 = df_total.loc[1:32].sum(axis=0)
next_token_1to64 = df_total.loc[1:64].sum(axis=0)
df_total.loc['total_token_time', :] = total_token_time
df_total.loc['next_token_time', :] = next_token_time
df_total.loc['next_token_1to16', :] = next_token_1to16
df_total.loc['next_token_1to32', :] = next_token_1to32
df_total.loc['next_token_1to64', :] = next_token_1to64
df_total_op_sum = df_total.sum(axis=1)
df_total.insert(0, 'TOTAL', df_total_op_sum)

print("\nper_sample_avg", df_total.divide(total_sample*inst_num), sep='\n')
df_op_percent = df_total.divide(df_total_op_sum, axis=0)
print("\ndf_op_percent", df_op_percent.mul(100).round(4).astype(str).add(' %'), sep='\n')

summary_rows = ['total_token_time', 0, 'next_token_time', 'next_token_1to16', 'next_token_1to32', 'next_token_1to64']
total_op = df_total.loc[summary_rows, 'TOTAL']
total_op = pd.DataFrame(total_op.rename('time'))
total_op.loc[:, 'percent'] = total_op.loc[:, 'time'].divide(total_op.loc['total_token_time', 'time'])
total_op.loc[:, 'percent'] = total_op.loc[:, 'percent'].mul(100).round(4).astype(str).add(' %')
print("\ntotal_op", total_op, sep='\n')

# first token => 26.8564
# next token => 73.1436
#                      => view 16.0243% => 0%
#                      => MUL_MAT+MUL_MAT_WITH_BIAS+MUL_QKV+FFN_ADD_GeLU 4.0026 % 1.5136 % 13.5369 % 37.6016 => 56.6547% * 1/2 => 28.32735%
#            => 73.1436 * (100-16.0243-28.32735)%=> 40.7032
# =====>
# new first:next=26.8564:40.7032
# old:new = 100:26.8564+40.7032 == 100:67.5596 => 148.02%
#
# new next token:
# non-attn:attn = 100-16.0243-28.32735 : 27.5969+2.0396 => 55.64835 : 29.6365 => 100 : 53.2567
