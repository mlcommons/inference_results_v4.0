# Steps to run sd-xl Server

### Environment setup 
To setup the environment follow the steps described in [/closed/Intel-HabanaLabs/code/README.md](../../../code/README.md)

### Commands
Run the following commands from [/closed/Intel-HabanaLabs/code/](../../../code/) directory.

#### Run accuracy
```bash
source stable-diffusion-xl/functions.sh
build_mlperf_inference --output-dir <output_dir> --submission sd-xl-bf16-full-8x_Server --mode acc
```

#### Run performance
```bash
source stable-diffusion-xl/functions.sh
build_mlperf_inference --output-dir <output_dir> --submission sd-xl-bf16-full-8x_Server --mode perf
```

### Results

You can find the logs under /output-dir/logs/sd-xl-bf16-full-8x/Server

For more details go to [/closed/Intel-HabanaLabs/code/README.md](../../../code/README.md)
