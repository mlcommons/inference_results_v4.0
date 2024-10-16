: ${is_in_container=${1:-"False"}}
: ${base_path=${2:-""}}
: ${output_path=${3:-"/data/mlperf_data"}}
: ${model=${4:-"na"}}
: ${impl=${5:-"na"}}
: ${dtype=${6:-"na"}}

f_output="${output_path}/env_${model}_${impl}_${dtype}.log"

if [ "${is_in_container}" == "False" ]; then
    pushd ${base_path}
    echo " git log --oneline | head -n 1:" > ${f_output}
    git log --oneline | head -n 1 >> ${f_output} || true
    echo "" >> ${f_output}
    popd

    echo "who:" >> ${f_output}
    who >> ${f_output} || true
    echo "" >> ${f_output}
    echo "free -h:" >> ${f_output}
    free -h >> ${f_output} || true
    echo "" >> ${f_output}
    echo "ps -ef | grep python:" >> ${f_output}
    ps -ef | grep python >> ${f_output} || true
    echo "" >> ${f_output}
    echo "lscpu:" >> ${f_output}
    lscpu >> ${f_output} || true
    dmesg | grep "cpu clock throttled" >> ${f_output} || true
    echo "" >> ${f_output}
else
    echo "conda list:" >> ${f_output}
    conda list >> ${f_output} || true
    echo "" >> ${f_output}
fi