#!/bin/bash

#. conda/bin/activate
# conda activate [가상환경 이름:team#] #임시(lss)


# sh run_tasks.sh 하면 안되고 bash run_tasks.sh 해야됨

cd scripts

python evaluate.py --pred="first.jsonl" --test="data.json"