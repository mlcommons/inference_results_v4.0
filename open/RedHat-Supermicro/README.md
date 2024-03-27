# MLPerf Inference v4.0 Red Hat Inc & Supermicro Submission


This is a repository for Supermicro & Red Hat's submission to the MLPerf Inference v4.0 benchmark.  It
includes implementations of the benchmark code optimized for running on Supermicro AS-4125GS-TNRT servers with NVIDIA 8XH100 GPUs.  

# Contents

Each model implementation in the `code` subdirectory has:
 
* Code that implements inferencing  
* A Dockerfile which can be used to build a container for the benchmark
* Documentation on the dataset, model, and machine setup

# Hardware & Software requirements

These benchmarks have been tested on the following machine configuration:

* A Supermicro GPU A+ Server, the AS-4125GS-TNRT ; with NVIDIA 8X H100 GPUS directly attached 
* The required software stack includes:
    - [Red Hat OpenShift](https://access.redhat.com/documentation/en-us/openshift_container_platform/4.14/html/installing/index)
    - [Node Feature Discovery Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-01.html)
    - [Nvidia GPU Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-03.html)
    - [OpenShift AI Model Serving Stack](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.5/html/working_on_data_science_projects/serving-large-language-models_serving-large-language-models)

Each benchmark can be run with the following steps:

1. Follow the instructions in the README in the api-endpoint-artifacts directory (which includes creating a pod to run the model inferencing)
2. Use the OpenShift console to observe resource ultilzation (e.g. GPU utilization) for model inferencing.
