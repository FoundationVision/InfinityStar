#!/usr/bin/env bash
set -x

####### Configure different distributed environment variables according to your platform.
# nccl setting
unset NCCL_NET_PLUGIN
unset NCCL_FASTRAK_ENABLE
unset NCCL_FASTRAK_USE_SNAP
unset NCCL_FASTRAK_NUM_FLOWS
unset NCCL_FASTRAK_*
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_NET_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_PLUGIN=NONE
export NCCL_NVLS_ENABLE=0


# arnold setting
unset ARNOLD_WORKER_0_HOST

TORCHRUN_RDZV_READ_TIMEOUT=${TORCHRUN_RDZV_READ_TIMEOUT:-600}
ARNOLD_ID=${ARNOLD_ID:-0}
ARNOLD_WORKER_0_HOST=${ARNOLD_WORKER_0_HOST:-'localhost'}
ARNOLD_WORKER_0_PORT=${ARNOLD_WORKER_0_PORT:-'9591'}
RUN_COMMAND=${@:2}
PORT=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
########---------------------------------------------------------------------------------------------------




# torch setting
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=infinity/models/:$PYTHONPATH
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=8

wandb offline
# wandb online
exp_name=debug_release
bed_path=./checkpoints/${exp_name}/
video_data_path='./data/infinitystar_toy_data/split_jsonls'

checkpoints_dir='./'
t5_path="${checkpoints_dir}text_encoder/flan-t5-xl-official"
vae_path="${checkpoints_dir}infinitystar_videovae.pth"
resume_path="${checkpoints_dir}infinitystar_8b_720p_weights"
vae_type=64
videovae=10
token_cache_dir=./checkpoints/local_cache/cached_visual_tokens_720p

local_out_path=$LOCAL_OUT/${exp_name}
video_fps=16
video_frames=81

LOCAL_OUT=checkpoints
mkdir -p $LOCAL_OUT

# 创建noise_apply_strength列表
noise_apply_strength=()
noise_apply_strength+=($(printf "0.3\n%.0s" {1..200}))
noise_apply_strength_str=$(IFS=,; echo "${noise_apply_strength[*]}")

torchrun --nproc_per_node=$ARNOLD_WORKER_GPU \
--nnodes=$ARNOLD_WORKER_NUM \
--master_addr=$ARNOLD_WORKER_0_HOST \
--node_rank=$ARNOLD_ID \
--master_port=$PORT \
--rdzv_conf=read_timeout=$TORCHRUN_RDZV_READ_TIMEOUT \
train.py \
--local_out_path ${local_out_path} \
--bed=${bed_path} \
--data_path=${image_data_path} \
--video_data_path=${video_data_path} \
--t5_path=${t5_path} \
--vae_type=${vae_type} \
--videovae=${videovae} \
--vae_path=${vae_path}  \
--token_cache_dir=${token_cache_dir} \
--tlr=4e-5 \
--pn=0.90M \
--model=infinity_qwen8b \
--project_name=infinity \
--exp_name=${exp_name} \
--checkpoint_type='torch' \
--enable_checkpointing=full-block \
--video_fps=${video_fps} \
--video_frames=${video_frames} \
--short_cap_prob 0.3 \
--use_streaming_dataset=1 \
--iterable_data_buffersize=1000 \
--enable_dynamic_length_prompt=1 \
--reweight_loss_by_scale 4 \
--zero=3 \
--save_model_iters_freq 200 \
--noise_apply_strength="$noise_apply_strength_str" \
--dynamic_scale_schedule=infinity_elegant_clip20frames_v2 \
--mask_type=infinity_elegant_clip20frames_v2 \
--use_flex_attn=True \
--use_vae_token_cache=1 \
--cache_check_mode=1 \
--allow_online_vae_feature_extraction=0 \
--train_with_var_seq_len=1 \
--video_var_len_prob='[60, 20, 10, 6, 2, 1, 1]' \
--drop_long_video=0 \
--image_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]' \
--video_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]' \
--append_duration2caption=1 \
--wp_it=0 \
--use_two_stage_lfq=1 \
--semantic_scale_dim=16 \
--detail_scale_min_tokens=750 \
--semantic_scales=12 \
--allow_less_one_elem_in_seq=1 \
--use_feat_proj=2 \
--restrict_data_size=-1 \
--enable_hybrid_shard=0 \
--sp_size=1 \
--drop_720p_last_scale=1 \
--torchshard_resume=${resume_path} \

# The largest scale of 720p has a long sequence length, which requires sequence parallel for multi-node training.
# --drop_720p_last_scale=0 \  
# --enable_hybrid_shard=1 \
# --sp_size=2 \