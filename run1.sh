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
CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "base" \
    --batch_size=32 --lr=2e-5 --pretrained="monologg/koelectra-base-v3-discriminator" \
    --seed=21

# run_task 1 CD
# CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "base" \
#     --batch_size=32 --lr=2e-5 --pretrained="kykim/electra-kor-base" \
#     --seed=21

# run_task 1 CD
# CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "base" \
#     --batch_size=32 --lr=2e-5 --pretrained="tunib/electra-ko-base" \
#     --seed=21

# run_task 1 CD
# CUDA_VISIBLE_DEVICES=1 python CD_pipeline.py --name "2CLS" \
#     --batch_size=32 --lr=2e-5 --pretrained="beomi/KcELECTRA-base" \
#     --seed=1111 --weight_decay=0.001
# 이건 clean 전처리 함수 적용 안함 아직
