sudo sh -c 'echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo'
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
sudo sh -c 'echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct'
sudo sh -c 'echo 3 >/proc/sys/vm/drop_caches'
sudo cpupower frequency-set -g performance
