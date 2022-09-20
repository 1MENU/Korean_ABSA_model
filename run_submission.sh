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

# # task1 : COLA
# run_task 1 COLA
# python COLA_submission.py --model "a"
# python COLA_submission.py --model "ars2"


# # task2 : WiC
# run_task 2 WiC
# python WIC_submission.py --model "a"
# python WIC_submission.py --model "ars2"


# task3 : COPA
run_task 3 COPA
python COPA_submission.py --model c
python COPA_submission.py --model cf


# # task4 : BoolQ
# run_task 4 BoolQ
# python BoolQ_submission.py --model d-0
# python BoolQ_submission.py --model d-1




cd ..

# enssemble


# # # Format Check
# format_check materials/submission/cola/aa-0.json