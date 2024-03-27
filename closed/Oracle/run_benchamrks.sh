# Formal runs

export OMPI_ALLOW_RUN_AS_ROOT=1 
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export LOG_DIR=/work/build/logs/OCI-H100-v4.0.3/Performance

make run RUN_ARGS="--benchmarks=dlrm-v2,llama2-70b,resnet50,retinanet,rnnt,bert,gptj,stable-diffusion-xl --scenarios=offline,server --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=dlrm-v2,llama2-70b,bert,gptj --config_ver=high_accuracy --scenarios=offline,server --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=3d-unet --config_ver=high_accuracy --scenarios=offline --test_mode=PerformanceOnly"

export LOG_DIR=/work/build/logs/OCI-H100-v4.0.3/Accuracy
make run RUN_ARGS="--benchmarks=dlrm-v2,llama2-70b,resnet50,retinanet,rnnt,bert,gptj,stable-diffusion-xl --scenarios=offline,server --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=dlrm-v2,llama2-70b,bert,gptj --config_ver=high_accuracy --scenarios=offline,server --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=3d-unet --config_ver=high_accuracy --scenarios=offline --test_mode=AccuracyOnly"
