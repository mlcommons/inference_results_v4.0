# Intel-HabanaLabs MLPerf™ inference submission
This directory provides instructions to reproduce Intel-HabanaLabs's results for MLPerf™ inference submission.\
MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries.\
All rights reserved. Unauthorized use is strictly prohibited.

## Setup

Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment.

### Prepare Intel-HabanaLabs MLPerf Inference Container

```bash
mkdir -p /path/to/Habana
export HABANA_DIR=/path/to/Habana
```

This README is located in [code](./) directory corresponding to Intel-HabanaLabs submission. Download the whole [code](./) folder along with all subfolders and copy it under $HABANA_DIR

```bash
docker run --privileged --security-opt seccomp=unconfined \
           --name mlperf-habana -td                \
           -v /dev:/dev                            \
           --device=/dev:/dev                      \
           -v /sys/kernel/debug:/sys/kernel/debug  \
           -v /tmp:/tmp                            \
           -v $HABANA_DIR:/root/Habana/            \
           --cap-add=sys_nice --cap-add=SYS_PTRACE \
           --user root --workdir=/root --net=host  \
           --ulimit memlock=-1:-1 vault.habana.ai/gaudi-docker-mlperf/ver4.0/pytorch-installer-2.1.1:1.14.98-33
```
```bash
docker exec -it mlperf-habana bash
```
### Download Checkpoint
```bash
mkdir -p /mnt/weka/data/mlperf_inference/stable-diffusion-xl/stable_diffusion_fp32
pushd /mnt/weka/data/mlperf_inference/stable-diffusion-xl
sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/stable_diffusion_fp32 ./stable_diffusion_fp32 -P
popd
```

### Download Dataset (Optional)
To download dataset run the below command, build_mlperf_inference covers same functionality.
```bash
pushd /root/Habana/code/stable-diffusion-xl/stable-diffusion-xl/tools
./download-coco-2014.sh -n <number_of_workers>
popd
```
##  Reproduce Results
### Get Started
Install the requirements and build the latest loadgen.

```bash
cd /root/Habana/code
source stable-diffusion-xl/functions.sh
pip install -r stable-diffusion-xl/stable-diffusion-xl/requirements.txt
build_mlperf_inference
```
### Generate Results
**To generate full submission results, run the following command:**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8
```
The command produces results from accuracy and performance runs for both Offline and Server scenarios.
Logs can be found under /path_to_output_dir/logs/model/, e.g. /results/logs/sd-xl-fp8/


**To generate results for Offline and Server scenarios separately, run the following commands:**
```bash
source stable-diffusion-xl/functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8_Offline
```

```bash
source stable-diffusion-xl/functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8_Server
```
Logs can be found under /path_to_output_dir/logs/model/scenario/, e.g. /results/logs/sd-xl-fp8/Offline/

**To generate results for accuracy and performance separately, add ```--mode``` flag as in one of the following commands:**
```bash
source stable-diffusion-xl/functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8_Offline --mode acc
```
```bash
source stable-diffusion-xl/functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission sd-xl-fp8_Offline --mode perf
```

Logs can be found under /path_to_output_dir/logs/model/scenario/mode/, e.g. /results/logs/sd-xl-fp8/Offline/accuracy/

### Calibration steps (Optional)
This is for recreating the measurements on the calibration dataset which we later use to determine the scales.
```bash
pushd /root/Habana/code/stable-diffusion-xl/stable-diffusion-xl
bash ./tools/measure.sh
popd
```

## Performance Optimization with FP8 Flow
To optimize performance, we set heavy-performance ops to operate in FP8-143.
In the conversion to FP8-143 we use various values of exponent bias which are determined using a calibration dataset.
For each input processed, the UNET block is iteratively invoked 20 times. A more aggressive form is used for the first 18 steps; and a less aggressive one for the final 2 steps.

### Environment Variables
All necessary environmental variables are enabled by default.
