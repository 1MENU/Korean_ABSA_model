#!/bin/bash

#. conda/bin/activate
# conda activate [가상환경 이름:team#] #임시(lss)


# sh run_tasks.sh 하면 안되고 bash run_tasks.sh 해야됨

cd scripts

enssemble(){
    echo "run enssemble check !!"
    python enssemble.py --name "a" --task 'cola' --preds "last4+lr1e-5_bs128_5fold5-1.json" "last4+lr1e-5_bs128-1" --weights 1 1
}

# enssemble


# # task1 : COLA
# echo "run enssemble COLA !!"
# python ensemble.py --name "a" --task 'cola' --preds "a" "ars2" --weights 1 1 --label "task1_grammar/NIKL_CoLA_test_labeled_v2.tsv"


# # task2 : WiC
# echo "run ensemble WiC !!"
# python ensemble.py --name "##_cls+mul_trflip_rs1-2_tta_flip" --task 'wic' --preds "##_cls+mul_trflip_rs1-2_f" "##_cls+mul_trflip_rs1-2" --weights 0.878 0.870 --label "task2_homonym/NIKL_SKT_WiC_Test_labeled.tsv"


# task3 : COPA
echo "run enssemble COPA !!"
python ensemble.py --name "c+cf" --task 'copa' --preds "c" "cf" --weights 1 1 --label "task3_COPA/SKT_COPA_Test_labeled.tsv"


# # task4 : BoolQ
# echo "run enssemble BoolQ !!"
# python ensemble.py --name "d0+d1" --task 'boolq' --preds "d-0" "d-1" --weights 1 1 --label "task4_boolQA/SKT_BoolQ_Test_labeled.tsv"
