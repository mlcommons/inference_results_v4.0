#!/bin/sh

export SUBMITTER=Oracle
# **** TEST01 ****
# run compilance of Offline benchmarks below
for i in resnet50 retinanet rnnt bert 3d-unet dlrm-v2
do
   [ -e "audit.config" ] && rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST01/$i/audit.config .
   make run_audit_test01 RUN_ARGS="--benchmarks=$i --scenarios=offline"
done

# run compliance of 5 Server benchmarks
for i in resnet50 retinanet rnnt bert dlrm-v2 
do
   [ -e "audit.config" ] && rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST01/$i/audit.config .
   make run_audit_test01 RUN_ARGS="--benchmarks=$i --scenarios=server"
done

for i in bert 3d-unet dlrm-v2
do
   [ -e "audit.config" ] && rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST01/$i/audit.config .
   make run_audit_test01 RUN_ARGS="--benchmarks=$i --config_ver=high_accuracy --scenarios=offline"
done
for i in bert dlrm-v2
do
   [ -e "audit.config" ] && rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST01/$i/audit.config .
   make run_audit_test01 RUN_ARGS="--benchmarks=$i --config_ver=high_accuracy --scenarios=server"
done

# **** TEST04 ***

for i in resnet50 
do 
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST04/audit.config .
   make run_audit_test04 RUN_ARGS="--benchmarks=$i --scenarios=offline"
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST04/audit.config .
   make run_audit_test04 RUN_ARGS="--benchmarks=$i --scenarios=server"
done
   rm -f audit.config

# **** TEST05 ****
for i in resnet50 retinanet rnnt bert dlrm-v2
do
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST05/audit.config .
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=offline"
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST05/audit.config .
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=server"
done

for i in 3d-unet 
do
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST05/audit.config .
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=offline"
done

for i in bert dlrm-v2
do
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST05/audit.config .
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=offline --config_ver=high_accuracy"
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST05/audit.config .
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=server --config_ver=high_accuracy"
done

for i in 3d-unet
do
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST05/audit.config .
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=offline --config_ver=high_accuracy"
done

rm -f audit.config

# **** TEST06 ****

for i in llama2-70b
do
   rm -f audit.config
   cp /work/build/inference/compliance/nvidia/TEST06/audit.config .
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=offline"
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=server"
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=offline --config_ver=high_accuracy"
   make run_audit_test05 RUN_ARGS="--benchmarks=$i --scenarios=server --config_ver=high_accuracy"
done
