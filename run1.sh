# sh run_tasks.sh 하면 안되고 bash run_tasks.sh 해야됨

value=$(<api_key.txt)   # = 띄어쓰기 하면 안됨. 붙여서 써야 할당 제대로 됨
echo "$value"

run_task(){    # wandb 재로그인, echo
    echo "run task$1 : $2 !!  [cuda : 1]"
    wandb login --relogin $value
}

cd scripts

# --name에 띄어쓰기, "/" 사용 금지 : _(언더바) 사용 추천

# CUDA_VISIBLE_DEVICES=1

run_task 1 CD
CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "bilstm-1" \
    --batch_size=64 --lr=2e-5 --pretrained="monologg/koelectra-base-v3-discriminator" \
    --seed=11 --LS=0.01

CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "bilstm-1" \
    --batch_size=16 --lr=3e-6 --pretrained="klue/roberta-base" \
    --seed=1

run_task 1 CD
CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "bilstm-1" \
    --batch_size=64 --lr=1e-5 --pretrained="monologg/koelectra-base-v3-discriminator" \
    --seed=11 --LS=0.01
    
CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "bilstm-1" \
    --batch_size=32 --lr=3e-6 --pretrained="klue/roberta-base" \
    --seed=1

# run_task 1 CD
# CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name="base_1linear" \
#     --batch_size=16 --lr=3e-6 --pretrained="klue/roberta-large" \
#     --seed=1



# 기본 test hyperparameter

# CUDA_VISIBLE_DEVICES=0 python CD_pipeline.py --name="base_model" \
#     --batch_size=16 --lr=3e-6 --pretrained="klue/roberta-large" \
#     --seed=1
# CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "base_model" \
#     --batch_size=64 --lr=2e-5 --pretrained="monologg/koelectra-base-v3-discriminator" \
#     --seed=11 --LS=0.01