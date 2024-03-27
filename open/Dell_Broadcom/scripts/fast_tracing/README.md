A post-processing script to generate visualization results out of Mlperf fast tracing logs

# Creating a log file

WIP

Enable Mlperf fast tracing and redirect STDOUT into a file

# Usage

Options:


```sh
usage: mlperf_perf_tr.py [-h] -i INPUT [-o OUTPUT] [-v VIEW]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input log file
  -o OUTPUT, --output OUTPUT
                        Json trace file
  -v VIEW, --view VIEW  Visualize latencies (gui|svg|png)
```

# Visualize latencies across time

Example of command line:

GUI:

```sh
./mlperf_perf_tr.py -i log.PERFTR.rn50.H100-PCIE-x8.success.txt -v gui
```

To file:

```
./mlperf_perf_tr.py -i log.PERFTR.rn50.H100-PCIE-x8.success.txt -v output.png
```

# Generate perfetto trace

Example of command line:

```sh
./mlperf_perf_tr.py -i log.PERFTR.rn50.H100-PCIE-x8.success.txt -o log.PERFTR.rn50.H100-PCIE-x8.perfetto.json
```

# How to use perfetto

Go to:

https://ui.perfetto.dev/

And select `Open trace file`

