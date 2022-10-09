# sh run_tasks.sh 하면 안되고 bash run_tasks.sh 해야됨

value=$(<api_key.txt)   # = 띄어쓰기 하면 안됨. 붙여서 써야 할당 제대로 됨
echo "$value"

run_task(){    # wandb 재로그인, echo
    echo "run task$1 : $2 !!  [cuda : 0]"
    wandb login --relogin $value
}

cd scripts

# --name에 띄어쓰기, "/" 사용 금지 : _(언더바) 사용 추천

# CUDA_VISIBLE_DEVICES=0


run_task 1 CD
CUDA_VISIBLE_DEVICES=0 python CD_pipeline.py --name "pf_biLstm-1" \
    --batch_size=16 --lr=8e-6 --pretrained="kykim/funnel-kor-base" \
    --seed=21 --save=1

# run_task 1 CD
# CUDA_VISIBLE_DEVICES=0 python CD_pipeline.py --name "1lay" \
#     --batch_size=16 --lr=7e-6 --pretrained="kykim/funnel-kor-base" \
#     --LS=0.01 --seed=11 --save=1