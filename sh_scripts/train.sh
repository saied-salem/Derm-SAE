#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/../..

seed_number=777

dataset='HAM10000' # HAM10000, BCCD, DTR

use_img_norm='false'
use_txt_norm='false'
use_residual='false'

func='our_main'
lr=0.0005
clip_model='ViT-L/14' 

cfn="our_selection" 
batch_size=256 # 256

concept_path="./gpt_concepts/${dataset}.json"
concept2type='false'
pearson_weight=0.9


if [ ${dataset} = "HAM10000" ]
then
    num_concept=70 # 7*50 HAM
    epoch=300 # 300
    num_layers=2
elif [ ${dataset} = "BCCD" ]
then
    num_concept=40 # 4*50 BCCD
    epoch=100 # 100
    num_layers=1
elif [ ${dataset} = "DTR" ]
then
    num_concept=250 # 5*50 DTR
    epoch=300 # 300
    num_layers=1
fi

if [ ${num_layers} = 1 ]
then
    name_num_layers='one' # 256
elif [ ${num_layers} = 2 ]
then
    name_num_layers='two' # 256
elif [ ${num_layers} = 4 ]
then
    name_num_layers='four' # 256
fi

directory="${dataset}testUtility"

echo "${cfn} ${seed_number} ${func} ${num_concept} ${directory}"

python Ada_cbm_simple.py --config cfg/HAM10000/HAM10000_allshot_fac.py \
--work_dir 'exp/HAM10000/HAM10000testUtility'

echo "${cfn} ${seed_number} ${func} ${num_concept} finished"
