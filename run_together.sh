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

CUDA_VISIBLE_DEVICES=0 python together.py \
    --cd="fCp_1lay_256_1e-05_N_2F3_rs2_kykE" \
    --sc="last4_32_2e-05_KcE_rs1111" \
    --name="7"

CUDA_VISIBLE_DEVICES=0 python together.py \
    --cd="fCp_1lay_256_2e-05_N_3F3_rs3_kykE" \
    --sc="last4_32_2e-05_KcE_rs1111" \
    --name="8"