CPUS_PER_INSTANCE=4
export OMP_NUM_THREADS=4

export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

number_threads=`nproc --all`
export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')
num_instance=$((number_cores/CPUS_PER_INSTANCE))
num_workers=$((CPUS_PER_INSTANCE/OMP_NUM_THREADS))


python ../../user_config.py
USER_CONF=user.conf

bash run_mlperf.sh --type=$1 \
	           --precision=int8 \
		   --user-conf=${USER_CONF} \
		   --num-instance=$num_instance \
		   --cpus-per-instance=$CPUS_PER_INSTANCE \
                   --workers=${num_workers} \
                   --scenario=Offline
