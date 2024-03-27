export NUM_SOCKETS=2        # i.e. 8
export CPUS_PER_SOCKET=60   # i.e. 28
export CPUS_PER_PROCESS=60  # which determine how much processes will be used
                            # process-per-socket = CPUS_PER_SOCKET/CPUS_PER_PROCESS
export CPUS_PER_INSTANCE=2  # instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                             # total-instance = instance-per-process * process-per-socket
export CPUS_FOR_LOADGEN=1   # number of cpus for loadgen
                            # finally used in our code is max(CPUS_FOR_LOADGEN, left cores for instances)
export BATCH_SIZE=400
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX