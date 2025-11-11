# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import time
import hashlib
import shutil
import re

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
from transformers import AutoTokenizer
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast
from timm.models import create_model
import imageio


from infinity.models.infinity import Infinity
from infinity.utils.load import load_visual_tokenizer
from infinity.models.basic import *
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor

def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val

def encode_prompt(t5_path, text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, low_vram_mode=False):
    if enable_positive_prompt:
        pass
    print(f'prompt={prompt}')
    captions = [prompt]
    if 'flan-t5' in t5_path:
        tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.cuda(non_blocking=True)
        mask = tokens.attention_mask.cuda(non_blocking=True)
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
        lens: List[int] = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)    
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    else:
        text_features = text_encoder(captions, 'cuda')
        lens = [len(item) for item in text_features]
        cu_seqlens_k = [0]
        for len_i in lens:
            cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
        Ltext = max(lens)
        kv_compact = torch.cat(text_features, dim=0).float()
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

def gen_one_example(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    input_use_interplote_up=False,
    low_vram_mode=False,
    args=None,
    get_visual_rope_embeds=None,
    context_info=None,
    noise_list=None,
    return_summed_code_only=False,
    mode='',
    former_clip_features=None,
    first_frame_features=None,
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, prompt, enable_positive_prompt, low_vram_mode=low_vram_mode)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, negative_prompt, low_vram_mode=low_vram_mode)
    else:
        negative_label_B_or_BLT = None
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        out = infinity_test.autoregressive_infer(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            input_use_interplote_up=input_use_interplote_up,
            low_vram_mode=low_vram_mode,
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            context_info=context_info,
            noise_list=noise_list,
            return_summed_code_only=return_summed_code_only,
            mode=mode,
            former_clip_features=former_clip_features,
            first_frame_features=first_frame_features,
        )
        if return_summed_code_only:
            return out
        else:
            pred_multi_scale_bit_labels, img_list = out
            
    print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    img = img_list[0]
    return img, pred_multi_scale_bit_labels

def get_prompt_id(prompt):
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    # ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file

def load_tokenizer(t5_path =''):
    print(f'[Loading tokenizer and text encoder]')
    if 'flan-t5' in t5_path:
        from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
        # text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16)
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
        text_encoder.to('cuda')
        text_encoder.eval()
        text_encoder.requires_grad_(False)
    else:
        raise ValueError(f'Not support t5_path: {t5_path}')
    return text_tokenizer, text_encoder

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)


def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    
    print(f'[Loading Infinity]')
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = create_model(
            args.model_type,
            vae_local=vae, text_channels=args.text_channels, text_maxlen=512,
            raw_scale_schedule=None,
            checkpointing='full-block',
            pad_to_multiplier=128,
            use_flex_attn=args.use_flex_attn,
            add_lvl_embeding_on_first_block=0,
            num_of_label_value=args.num_of_label_value,
            rope2d_each_sa_layer=args.rope2d_each_sa_layer,
            rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
            pn=args.pn,
            apply_spatial_patchify=args.apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[0.571, 1.0],
            video_frames=args.video_frames,
            other_args=args,
        ).to(device=device)
        print(f'[you selected Infinity with {args.model_type}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={args.bf16}')
        if args.bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()
        infinity_test.eval()
        infinity_test.requires_grad_(False)
        infinity_test.cuda()
        torch.cuda.empty_cache()

        if not model_path:
            return infinity_test
        
        print(f'============== [Load Infinity weights] ==============')    
        if args.checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(infinity_test.load_state_dict(state_dict))
        elif args.checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        elif args.checkpoint_type == 'omnistore':
            from infinity.utils.save_and_load import merge_ckpt
            if args.enable_model_cache and osp.exists(args.cache_dir):
                local_model_dir = osp.abspath(osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_')))
            else:
                local_model_dir = osp.abspath(model_path)
            print(f'load checkpoint from {local_model_dir}')
            state_dict = merge_ckpt(local_model_dir, osp.join(local_model_dir, 'ouput'), save=False, fsdp_save_flatten_model=args.fsdp_save_flatten_model)
            print(infinity_test.load_state_dict(state_dict))
        infinity_test.rng = torch.Generator(device=device)
    return infinity_test

def images2video(ndarray_image_list, fps=24, save_filepath='tmp.mp4'):
    if len(ndarray_image_list) == 1:
        save_filepath = save_filepath.replace('.mp4', '.jpg')
        cv2.imwrite(save_filepath, ndarray_image_list[0])
        print(f"Image saved as {osp.abspath(save_filepath)}")
    else:
        h, w = ndarray_image_list[0].shape[:2]
        os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        imageio.mimsave(save_filepath, ndarray_image_list[:, :, :, ::-1], fps=fps,)
        print(f"Video saved as {osp.abspath(save_filepath)}")

def read_video_as_frames(video_path):
    if video_path.endswith('.jpg'):
        return cv2.imread(video_path)[None, ...]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return None
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    cap.release()
    frames = np.stack(frames)
    return frames

def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '0.40M', '0.90M'])
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=64)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_on_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--num_of_label_value', type=int, default=2)
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument('--dynamic_scale_schedule', type=str, default='13_hand_craft')
    parser.add_argument('--video_frames', type=int, default=81)
    parser.add_argument('--videovae', type=int, default=10)
    parser.add_argument('--fake_vae_input', type=int, default=0, choices=[0,1])
    parser.add_argument('--casual_multi_scale', type=int, default=0, choices=[0,1])
    parser.add_argument('--scale_embeds_num', type=int, default=128)
    parser.add_argument('--train_h_div_w_list', type=float, default=None, nargs='+')
    parser.add_argument('--mask_type', type=str, default='infinity_elegant_clip20frames_v2')
    parser.add_argument('--context_frames', type=int, default=1000)
