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

python label.py
