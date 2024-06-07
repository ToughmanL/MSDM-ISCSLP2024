#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1"


stage=1
stop_stage=1

# The num of nodes
num_nodes=1
# The rank of current node
node_rank=0

# MSDM training set
train_set=train
dev_set=dev
test_sets=test

# resnet_10 conformer_3drop4 conformer3_resnet10 conformer4_resnet18
model_name=conformer4_resnet18
train_config=conf/train_${model_name}.yaml

cmvn=true
cmvn_sampling_divisor=20 # 20 means 5% of the training data to estimate cmvn
dir=exp/${model_name}
mkdir -p $dir
data_type="raw"

# checkpoint=exp/${model_name}/0.pt
checkpoint=
decode_checkpoint=
average_num=2

. tools/parse_options.sh || exit 1;

set -u
set -o pipefail


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Compute cmvn"
  # 此处计算cmvn，如果无法创建wav.scp，可使用examples/MSDM/local/datalist2scp.py
  if $cmvn; then
    full_size=`cat data/${train_set}/wav.scp | wc -l`
    sampling_size=$((full_size / cmvn_sampling_divisor))
    shuf -n $sampling_size data/$train_set/wav.scp \
      > data/$train_set/wav.scp.sampled
    python3 tools/compute_cmvn_stats.py \
    --num_workers 32 \
    --train_config $train_config \
    --in_scp data/$train_set/wav.scp.sampled \
    --out_cmvn data/$train_set/global_cmvn \
    || exit 1;
  fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Start training"
  mkdir -p $dir
  # INIT_FILE is for DDP synchronization
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    rank=`expr $node_rank \* $num_gpus + $i`
    python wenet/bin/train_dys.py --gpu $gpu_id \
      --config $train_config \
      --data_type $data_type \
      --train_data data/$train_set/data.list \
      --cv_data data/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      $cmvn_opts \
      --num_workers 8 \
      --pin_memory
  } &
  done
  wait
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  decode_checkpoint=$dir/avg${average_num}.pt
  if [ ! -f $decode_checkpoint ]; then
    echo "average model"
    python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --val_best
  fi
  for test_set in ${test_sets}; do
    test_dir=$dir/${test_set}
    mkdir -p $test_dir
    python wenet/bin/infer_dys.py --gpu 1 \
      --config $dir/train.yaml \
      --data_type raw \
      --test_data data/${test_set}/data.list \
      --checkpoint $decode_checkpoint \
      --result_dir $test_dir \
      --batch_size 8
  done
fi

echo "Done"

