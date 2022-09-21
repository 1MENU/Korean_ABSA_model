# !/bin/bash

# . conda/bin/activate
# conda activate team2


# sh run_tasks.sh 하면 안되고 bash run_tasks.sh 해야됨

value=$(<api_key.txt)   # = 띄어쓰기 하면 안됨. 붙여서 써야 할당 제대로 됨
echo "$value"

run_task(){    # wandb 재로그인, echo
    echo "run task$1 : $2 !!"
    wandb login --relogin $value
}

cd scripts

# --name에 띄어쓰기, "/" 사용 금지 : _(언더바) 사용 추천

# python baseline.py \
#   --train_data ../dataset/train1.jsonl \
#   --dev_data ../dataset/dev1.jsonl \
#   --base_model xlm-roberta-base \
#   --do_train \
#   --do_eval \
#   --learning_rate 3e-6 \
#   --eps 1e-8 \
#   --num_train_epochs 20 \
#   --entity_property_model_path ../saved_model/category_extraction/ \
#   --polarity_model_path ../saved_model/polarity_classification/ \
#   --batch_size 8 \
#   --max_len 256


# python baseline.py \
#   --test_data ../dataset/dev.jsonl \
#   --base_model xlm-roberta-base \
#   --do_test \
#   --entity_property_model_path ../saved_model/category_extraction/saved_model_epoch_1.pt \
#   --polarity_model_path ../saved_model/polarity_classification/saved_model_epoch_2.pt \
#   --batch_size 16 \
#   --max_len 256


run_task 1 CD
python CD_pipeline.py --epochs 30 --name test_1 --batch_size=16 --lr=3e-6
