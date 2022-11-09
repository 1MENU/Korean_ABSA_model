# sh run_tasks.sh 하면 안되고 bash run_tasks.sh 해야됨

value=$(<api_key.txt)   # = 띄어쓰기 하면 안됨. 붙여서 써야 할당 제대로 됨
echo "$value"

run_task(){    # wandb 재로그인, echo
    echo "run task$1 : $2 !!  [cuda : 0]"
    # wandb login --relogin $value
}

cd scripts

# --name에 띄어쓰기, "/" 사용 금지 : _(언더바) 사용 추천

# CUDA_VISIBLE_DEVICES=0

# --nsplit=3 --kfold=1


run_task 2 SC
CUDA_VISIBLE_DEVICES=0 python SC_pipeline.py --name "A2" \
    --batch_size=64 --lr=4e-5 --pretrained="kykim/electra-kor-base" \
    --LS=0 --weight_decay=0.1 --seed=817 --save=1 --nsplit=3 --kfold=3
