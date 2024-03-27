#!/bin/bash

srun --container-image= {Base Container Image}\
	--container-mounts  {Container Mounts}\
	--container-workdir {Path to Container Working Directory} \
    python main.py --engine-dir= {Path to Engine Directory} \
    --model-path= {Path to model} \
    --dataset-path= {Path to dataset}\
    --mlperf-conf= {Path to mlperf.conf} \
    --user-conf= {Path to user.conf} \
    --scenario {Offline, Server} \
    -- {accuracy, performance}
