# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import sys
import json
import argparse
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))

import cv2
import torch
import random
import shutil
import numpy as np

from tools.run_infinity import *
from infinity.utils.video_decoder import EncodedVideoOpencv
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func


def tensor2images(tensor):
    """Convert [bs,3,t,h,w] tensor to list of np.uint8 images
    """
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.permute(0,2,3,4,1) # [bs, 3, t, h, w] -> [bs, t, h, w, 3]
    tensor = tensor.mul_(255).to(torch.uint8).flip(dims=(4,))
    tensor = tensor.cpu().numpy()
    return tensor

if __name__ == '__main__':
    args=argparse.Namespace(
        pn='0.40M',
        fps=16,
        model_type='infinity_qwen8b',
        h_div_w_template=1.000,
        cache_dir='/dev/shm',
        seed=0,
        bf16=0,
        temporal_slice=0,
        enable_model_cache=0,
        scale_embeds_num=128,
        train_h_div_w_list=[0.571, 1.0],
        steps_per_frame=3,
        context_frames=1000,
        image_batch_size=1,
        video_batch_size=1,
        down_size_limit=340,
        casual_multi_scale=0,
        noise_apply_layers=200,
        noise_apply_requant=1,
        noise_apply_strength=[0. for _ in range(100)],
        video_caption_type='tarsier2_caption',
        temporal_compress_rate=4,
        cached_video_frames=81,
        learn_residual=0,
        use_diffloss=0,
        diffusion_batch_mul=0,
        video_fps=16,
        power_value=1.0,
        noise_apply_random_one=0,
        inject_sync=0,
        scales_256=11,
        dummy_text_len_in_seq=0,
        scale_max_token_len=-1,
        same_batch_among_ranks=0,
        use_flex_attn=0,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        sampling_per_bits=1,
    )

    checkpoints_dir='./'
    args.model_path='[TBD]'
    args.vae_path=os.path.join(checkpoints_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt=os.path.join(checkpoints_dir, 'text_encoder/flan-t5-xl-official/')

    args.vae_path='/mnt/bn/foundation-vision/hanjian.thu123/VideoVAE/video_vae_results/hj_vae_l2c16_l2c64_8_semantic_reg_after_semantics/slim_vae/model_step_7799.ckpt'
    args.text_encoder_ckpt='/mnt/bn/foundation-vision/hanjian.thu123/data/flan-t5-xl-official'
    # args.model_path='/mnt/bn/foundation-vision/hanjian.thu123/infinity/star/vgpt/checkpoints_vision/infinity_star_multiple_clips_sp8_with_pretrain_2e4_revised_caption2_040m_sp_size4_100w_data_second_clip_only7_longcat_video_ref_first_frame/global_step_24000'
    # args.checkpoint_type='omnistore'
    args.model_path='/tmp/weights/infinity_interact_24k'
    args.checkpoint_type='torch_shard'
    

    args.set_motion_score = -1
    args.min_scale_ind=3
    args.loop_times_per_scale=1
    args.global_sid_pe=0
    args.h_div_w = 0.571
    args.input_noise=1
    args.use_cfg, args.use_apg, args.apg_norm_threshold = 1, 0, 0.15
    args.diffusion_steps=-1
    args.infinity_diffusion_sample_topk=1
    args.noise_input=0
    args.reduce_accumulate_error_method='bsc'
    args.map_to_wide_weights=0
    args.min_duration=-1
    args.use_space_time_quant=0
    args.use_learnable_dim_proj=0
    args.semantic_scale_dim=16
    args.detail_scale_dim=64
    args.use_prompt_engineering = False
    args.context_from_largest_no=1
    args.max_repeat_times=1000
    args.text_channels=2048
    args.dynamic_scale_schedule='infinity_star_interact'
    args.mask_type='infinity_star_interact'
    args.semantic_scales=11
    args.detail_scale_min_tokens=350
    args.video_frames=161
    args.max_duration=10
    args.videovae=10
    args.vae_type=64
    args.num_lvl=2
    args.num_of_label_value=args.num_lvl
    args.semantic_num_lvl=args.num_lvl
    args.semantic_scale_dim=16
    args.detail_num_lvl=args.num_lvl
    args.detail_scale_dim=64
    args.use_clipwise_caption=1
    args.use_prompt_engineering = False
    args.vae_detail='discrete_flow_vae'
    args.use_feat_proj=2
    args.use_fsq_cls_head=0
    args.rope_type = '4d'
    args.noise_apply_strength = 0.0
    args.task_type='t2v'
    args.inner_scale_boost=0
    args.append_duration2caption=1
    args.n_sampes=1
    args.duration_resolution=1
    args.frames_inner_clip=20
    args.image_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]'
    args.video_scale_repetition = args.image_scale_repetition
    # args.taui, args.tauv = 0.4, 0.4
    # args.use_cfg, args.use_apg, args.cfg, args.apg_norm_threshold = 0, 1, 30, 0.05
    args.taui, args.tauv = 0.5, 0.5
    args.use_cfg, args.use_apg, args.cfg, args.apg_norm_threshold = 1, 0, 3, 0.05
    args.tau = [args.taui] * len(json.loads(args.image_scale_repetition)) + [args.tauv] * len(json.loads(args.video_scale_repetition))
    args.context_interval=2
    args.simple_text_proj=1
    args.apply_spatial_patchify=0
    args.use_two_stage_lfq=1
    args.fsdp_save_flatten_model=1
    args.two_gpu_infer=False

    scale_repetition = ''
    gt_leak = -1
    quality_prompt = ''

    video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
    total_secs = (args.video_frames-1) / args.fps
    if args.two_gpu_infer:
        args.other_device = 'cuda:1'
    else:
        args.other_device = 'cuda'

    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    save_dir_root = osp.join('tmp_videos', osp.basename(osp.dirname(args.model_path)), osp.basename(args.model_path))
    save_name = f'pn{args.pn}_fps{args.fps}_elegant_overfit100_rep_vf{args.video_frames}_cinterval_{args.context_interval}_use_cfg_{args.cfg}_use_apg_{args.use_apg}_cfg{args.cfg}_apg_norm_thre_{args.apg_norm_threshold}_taui{args.taui:.1f}_tauv{args.tauv:.1f}_gt_leak_{gt_leak}'
    save_dir_root = osp.join(save_dir_root, save_name)
    if osp.exists(save_dir_root):
        shutil.rmtree(save_dir_root)
    
    print(args)
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))

    import json
    with open('/mnt/bn/foundation-vision/hanjian.thu123/infinity/star/vgpt/data/long_cat_video_interactive_data_train_test_splits/train.json', 'r') as f:
        story_dirs = json.load(f)

    test_data_dir = 'data/interactive_toy_data'

    for dir_ind, story_dir in enumerate(story_dirs):
        story_id = osp.basename(story_dir)
        os.system(f'cp -r {story_dir} {test_data_dir}')
        prompt_path = osp.join(story_dir, 'prompt.txt')
        with open(prompt_path, 'r') as f:
            prompts = f.readlines()
        prompts = [f'<<<t=5s>>>{item.strip()}' for item in prompts]
        first_frame_features = None
        for ind, prompt in enumerate(prompts):
            save_dir = osp.join(save_dir_root, f'{dir_ind:04d}_{story_id}')
            if ind == 0:
                mode = 'first_iv_clip'
                video = EncodedVideoOpencv(osp.join(story_dir, '0000_refine_720p.mp4'), '0000_refine_720p.mp4', num_threads=0)
                raw_video, _ = video.get_clip(video.duration-5, video.duration, 81)
                h, w, _ = raw_video[0].shape
                h_div_w_template_ = h_div_w_template_list[np.argmin(np.abs(h/w-h_div_w_template_list))]
                scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][21]
                vae_stride = 16
                tgt_h, tgt_w = scale_schedule[-1][1] * vae_stride, scale_schedule[-1][2] * vae_stride
                img_T3HW = [transform(Image.fromarray(frame[:,:,::-1]), tgt_h, tgt_w) for frame in raw_video]
                img_T3HW = torch.stack(img_T3HW, 0) # [t,3,h,w]
                img_bcthw = img_T3HW.permute(1,0,2,3).unsqueeze(0).to('cuda') # [c,t,h,w] -> [b,c,t,h,w]
                args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
                args.tower_split_index = args.first_full_spatial_size_scale_index + 1
                scales_in_one_clip = args.first_full_spatial_size_scale_index + 1
                cur_scale_schedule = scale_schedule[scales_in_one_clip:]
                context_info = get_scale_pack_info(cur_scale_schedule, args.first_full_spatial_size_scale_index, args)
                former_clip_features, _, _ = vae.encode_for_raw_features(img_bcthw, scale_schedule=None, slice=True)
                # recons first frame
                recons_video = vae.decode(former_clip_features, slice=True)
                recons_video = tensor2images(recons_video)
                ref_video_path = osp.join(save_dir, f"{ind:04d}.mp4")
                images2video(recons_video[0], fps=args.fps, save_filepath=ref_video_path)
                if first_frame_features is None:
                    first_frame_features = former_clip_features[:,:,0:1]
                raw_video = np.array([cv2.resize(img, (tgt_w, tgt_h)) for img in raw_video])
                ref_video_path = osp.join(save_dir, f"{ind:04d}_gt.mp4")
                images2video(raw_video, fps=args.fps, save_filepath=ref_video_path)
                shutil.copyfile(prompt_path, osp.join(save_dir, f"prompt.txt"))
            else:
                mode = 'second_v_clip'
                video, former_clip_features = gen_one_example(
                    infinity,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    prompt,
                    negative_prompt="",
                    g_seed=args.seed,
                    gt_leak=-1,
                    gt_ls_Bl=None,
                    cfg_list=args.cfg,
                    tau_list=args.tau,
                    scale_schedule=cur_scale_schedule,
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                    enable_positive_prompt=False,
                    low_vram_mode=True,
                    args=args,
                    get_visual_rope_embeds=get_visual_rope_embeds,
                    context_info=context_info,
                    noise_list=None,
                    mode=mode,
                    former_clip_features=former_clip_features,
                    first_frame_features=first_frame_features,
                )
                video = video.cpu().numpy()
                ref_video_path = osp.join(save_dir, f"{ind:04d}.mp4")
                images2video(video, fps=args.fps, save_filepath=ref_video_path)
