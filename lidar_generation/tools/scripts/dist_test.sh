#!/usr/bin/env bash

export PATH="/data/ericliu/my_packages/miniconda3/bin:$PATH" 
__conda_setup="$('/data/ericliu/my_packages/miniconda3/condabin/conda' 'shell.bash' 'hook' 2> /dev/null)" 
eval "$__conda_setup" 
conda activate occ2lidar

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} tools/test.py --launcher pytorch ${PY_ARGS}