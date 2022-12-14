# 실행 : bash run_train.sh

# value=$(<api_key.txt)   # = 띄어쓰기 하면 안됨. 붙여서 써야 할당 제대로 됨
# echo "$value"

run_task(){    # wandb 재로그인, echo
    echo "run task$1 : $2 !!  [cuda : 0]"
    # wandb login --relogin $value
}

cd scripts

# --name에 띄어쓰기, "/" 사용 금지 : _(언더바) 사용 추천

CUDA_VISIBLE_DEVICES=0 python together.py \
    --cd  \
    "A2_64_1e-05_N_1F3_rs776_kykE" \
    "A2_64_2e-05_N_2F3_rs830_kykE" \
    "A2_64_9e-06_N_3F3_rs702_kykE" \
    --sc \
    "A2_64_1e-05_N_1F3_rs1_kykE" \
    "A2_64_4e-05_N_3F3_rs817_kykE" \
    "A2_64_8e-06_N_2F3_rs1000_kykE" \
    --pretrained="kykim/electra-kor-base" \
    --test_file="nikluge-sa-2022-test.jsonl" \
    --name="final"