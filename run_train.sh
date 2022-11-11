# 실행 : bash run_train.sh

# value=$(<api_key.txt)
# echo "$value"

run_task(){    # wandb 재로그인, echo
    echo "run task$1 : $2 !!  [cuda : 0]"
    # wandb login --relogin $value
}

cd scripts


# --name에 띄어쓰기, "/" 사용 금지 : _(언더바) 사용 추천

run_task 1 CD
CUDA_VISIBLE_DEVICES=0 python CD_pipeline.py --name "CD_test" \
    --batch_size=64 --lr=1e-5 --pretrained="kykim/electra-kor-base" \
    --LS=0 --weight_decay=0.1 --seed=776 --save=1 --nsplit=3 --kfold=1

run_task 2 SC
CUDA_VISIBLE_DEVICES=0 python SC_pipeline.py --name "SC_test" \
    --batch_size=64 --lr=8e-06 --pretrained="kykim/electra-kor-base" \
    --LS=0.001 --weight_decay=0.001 --seed=1000 --save=1 --nsplit=3 --kfold=2