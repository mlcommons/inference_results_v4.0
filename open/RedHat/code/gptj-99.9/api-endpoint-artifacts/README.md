# Using OpenShift AI Model Serving (TGIS) with MLPerf Inference

Prerequisites:
 - Install the OpenShift AI model serving stack
 - Add your AWS credentials to `secret.yaml` access the model files
 - Apply `secret.yaml`, `sa.yaml`
 - FOR TGIS: Apply `serving-tgis.yaml`, then finally `model.yaml`
 - FOR VLLM: Apply `serving-vllm.yaml`, then finally `model-vllm.yaml`
 - Create a benchmark pod using `benchmark.yaml`

In the pod, before any benchmark, first run `cd inference/language/gpt-j`

## STANDALONE TGIS INSTRUCTIONS
For the full accuracy benchmark (offline), run in the pod:
```
python3 -u main.py --scenario Offline --model-path ${CHECKPOINT_PATH} --api-server <INSERT API HOST> --api-model-name gpt-j-cnn --mlperf-conf mlperf.conf --accuracy --vllm --user-conf user.conf --dataset-path ${DATASET_PATH} --output-log-dir offline-logs --dtype float32 --device cpu 2>&1 | tee offline_performance_log.log
```
You can then run the same evaluation/consolidation scripts as the regular benchmark

Example API host: `https://gpt-j-isvc-predictor-gpt-service.apps.gdr-perf.perf.eng.bos2.dc.redhat.com`


For the performance benchmark (offline), run in the pod:
```
python3 -u main.py --scenario Offline --model-path ${CHECKPOINT_PATH} --api-server <INSERT API HOST> --api-model-name gpt-j-cnn --mlperf-conf mlperf.conf --vllm --user-conf user.conf --dataset-path ${DATASET_PATH} --output-log-dir offline-logs --dtype float32 --device cpu 2>&1 | tee offline_performance_log.log
```
(It is the same, just with `--accuracy` removed)

 - For multiple endpoints, add `--additional-servers <server 1> <server 2> ...`


For the performance benchmark (server), run in the pod:
```
python3 -u main.py --scenario Server --model-path ${CHECKPOINT_PATH} --api-server <INSERT API HOST> --api-model-name gpt-j-cnn --mlperf-conf mlperf.conf --vllm --user-conf user.conf --dataset-path ${DATASET_PATH} --output-log-dir server-logs --dtype float32 --device cpu 2>&1 | tee server_performance_log.log
```
(Configure target qps in `user.conf`)


NOTE: Hyperparams are currently configured for N instance x H100 80GB
