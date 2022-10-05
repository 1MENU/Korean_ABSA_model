#!/bin/bash

#. conda/bin/activate
# conda activate [가상환경 이름:team#] #임시(lss)


# sh run_tasks.sh 하면 안되고 bash run_tasks.sh 해야됨


run_task(){    # wandb 재로그인, echo
    echo "run submission$1 : $2 !!"
}

format_check(){     # Format Check
    echo "run format check !! : $1"
    python format_check.py --predFile $1
}


cd scripts

# task1 : CD
run_task 1 CD
CUDA_VISIBLE_DEVICES=1 python CD_pred.py --model="pair,_1lay_16_8e-06_fun_rs21" --pretrained="kykim/electra-kor-base"

# kykim/electra-kor-base