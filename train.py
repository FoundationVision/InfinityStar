# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import gc
import json
import math
import os
import os.path as osp
import random
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from functools import partial
from distutils.util import strtobool
from typing import List, Optional, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = '1'

import numpy as np
import torch
torch._dynamo.config.cache_size_limit = 64
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
import torch.distributed as tdist

import infinity.utils.dist as dist
from infinity.dataset.build import build_joint_dataset
from infinity.utils.save_and_load import CKPTSaver, omnistoreCheckpoint, auto_resume, omnistore_auto_resume
from infinity.models.ema import get_ema_model
from infinity.utils import arg_util, misc, wandb_utils
from infinity.trainer import get_trainer
# from infinity.utils.mfu.mfu import mfutool

def build_everything_from_args(args: arg_util.Args, saver):
    # set seed
    args.set_initial_seed(benchmark=True)
    # build tokenizer
    print(f'Loading T5 from {args.t5_path}...')
    if 'flan-t5' in args.t5_path:
        from transformers import T5EncoderModel, T5TokenizerFast
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(args.t5_path, revision=None, legacy=True) # text_tokenizer.model_max_length is 512
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        args.text_tokenizer_type = 'flan_t5'
        args.text_tokenizer = text_tokenizer
    else: # umt5
        raise ValueError("Only flan-t5 is supported now.")

    # build models. Note that here gpt is the causal VAR transformer which performs next scale prediciton with text guidance
    vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(args)
    
    # IMPORTANT: import heavy package `InfinityTrainer` after the Dataloader object creation/iteration to avoid OOM
    InfinityTrainer = get_trainer(args)
    # build trainer
    trainer = InfinityTrainer(
        device=args.device, 
        raw_scale_schedule=args.scale_schedule,
        vae_local=vae_local, 
        gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp,
        gpt_opt=gpt_optim, 
        label_smooth=args.label_smooth, 
        zero=args.zero, 
        vae_type=args.vae_type,
        reweight_loss_by_scale=args.reweight_loss_by_scale, 
        gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, 
        use_fsdp_model_ema=args.use_fsdp_model_ema, 
        other_args=args,
    )
    
    # auto resume from broken experiment
    global_it = 0
    if args.checkpoint_type == 'torch':
        auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
        print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
        print(f'initial args:\n{str(args)}')
        args.dump_log()
        if start_ep == args.epoch:
            args.dump_log()
            print(f'[vgpt] AR finished ({acc_str}), skipping ...\n\n')
            return None
        if trainer_state is not None and len(trainer_state):
            trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    elif args.checkpoint_type == 'omnistore':
        resume_path, info = omnistore_auto_resume(args, 'global_step_*')
        if not resume_path and args.rush_omnistore_resume:
            resume_path = args.rush_omnistore_resume
        if resume_path:
            print(f"omnistore resume from {resume_path}", flush=True)
            args_state, start_ep, start_it, global_it, acc_str, eval_milestone = saver.load(resume_path, fsdp_object=trainer.gpt, optimizer_object=trainer.gpt_opt.optimizer)
            dist.barrier()
        if args.rush_omnistore_resume == resume_path:
            global_it = 0
        auto_resume_info, acc_str, eval_milestone, trainer_state, args_state =  info, '[no acc str]', [], {}, {}
        
    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return text_tokenizer, text_encoder, trainer, global_it


def build_model_optimizer(args):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from infinity.models.infinity import Infinity, MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.lr_control import filter_params
    from infinity.utils.load import build_vae_gpt
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, gpt_wo_ddp = build_vae_gpt(args, device=args.model_init_device)
    count_p = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    num_para = count_p(gpt_wo_ddp)
    if num_para/1000 < 20: # < 20B
        gpt_wo_ddp = gpt_wo_ddp.to('cuda')

    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    init_weights(gpt_wo_ddp, other_std=args.tini)
    gpt_wo_ddp.special_init()
    if args.use_fsdp_model_ema:
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    else:
        gpt_wo_ddp_ema = None
    
    if args.rush_resume:
        print(f"{args.rush_resume=}")
        cpu_d = torch.load(args.rush_resume, 'cpu')
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
            ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
        else:
            state_dict = cpu_d
            ema_state_dict = state_dict
        def drop_unfit_weights(state_dict):
            if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                print(f'[rush_resume] drop word_embed.weight')
                del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != gpt_wo_ddp.head.out_features):
                print(f'[rush_resume] drop head.weight')
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != gpt_wo_ddp.head.bias.shape[0]):
                print(f'[rush_resume] drop head.bias')
                del state_dict['head.bias']
            if 'text_proj_for_sos.ca.mat_kv.weight' in state_dict and \
                (state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape):
                print(f'[rush_resume] drop cfg_uncond')
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            if 'semantic_head.weight' in state_dict:
                print(f'[rush_resume] replace semantic_head with semantic_head2')
                state_dict['semantic_head2.weight'] = state_dict['semantic_head.weight']
                state_dict['semantic_head2.bias'] = state_dict['semantic_head.bias']
                del state_dict['semantic_head.weight']
                del state_dict['semantic_head.bias']
            if 'semantic_head2.weight' in state_dict and (state_dict['semantic_head2.weight'].shape[0] != gpt_wo_ddp.semantic_head2.out_features):
                print(f'[rush_resume] drop semantic_head2.weight, semantic_head2.bias')
                del state_dict['semantic_head2.weight']
                del state_dict['semantic_head2.bias']
            return state_dict
        print(gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False))
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)
    elif args.torchshard_resume:
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(gpt_wo_ddp, args.torchshard_resume, strict=False)

    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    print(f'[PT][#para], GPT={num_para:.2f}\n\n')
    
    gpt_uncompiled = gpt_wo_ddp

    gpt_ddp_ema = None
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh

        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if gpt_wo_ddp.num_block_chunks == 1:  # no chunks
            auto_wrap_policy = ModuleWrapPolicy([type(gpt_wo_ddp.unregistered_blocks[0]), ])
        else:
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])
        
        if args.enable_hybrid_shard:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            world_size = dist.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree < world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)

        if args.fsdp_init_device == 'cpu':
            gpt_wo_ddp = gpt_wo_ddp.cpu()

        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
            device_mesh=device_mesh,
        ).to(args.device)
        
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema, 
                device_id=dist.get_local_rank(),
                sharding_strategy=sharding_strategy, 
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy, 
                use_orig_params=args.fsdp_orig, 
                sync_module_states=True, 
                limit_all_gathers=True,
            )
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== build optimizer ===============
    nowd_keys = set()
    if args.disable_weight_decay:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
    names, paras, para_groups = filter_params(gpt_ddp if args.zero else gpt_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    if '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1
    
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.fused_adam),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.fused_adam),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.adam_eps: opt_kw['eps'] = args.adam_eps
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = AmpOptimizer('gpt', args.fp16, opt_clz(params=para_groups, **opt_kw), gpt_ddp if args.zero else gpt_wo_ddp, args.r_accu, args.grad_clip, args.zero)
    del names, paras, para_groups
    return vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim


def build_dataset(args):
    train_dataset = build_joint_dataset(
        args, 
        args.data_path,
        args.video_data_path,
        max_caption_len=args.tlen, 
        short_prob=args.short_cap_prob, 
        load_vae_instead_of_image=False
    )
    return train_dataset

def main_train(args: arg_util.Args):
    if args.checkpoint_type == 'torch':
        saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    elif args.checkpoint_type == 'omnistore':
        saver = omnistoreCheckpoint(eval_milestone=None)
    else:
        raise ValueError(f'{args.checkpoint_type=}')
    ret = build_everything_from_args(args, saver)
    
    if ret is None:
        return
    
    text_tokenizer, text_encoder, trainer, start_global_it = ret
    gc.collect(), torch.cuda.empty_cache()
    seg5 = np.linspace(1, args.epoch, 5+1, dtype=int).tolist()
    
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, args.epoch // 10) if args.epoch <= 100 else max(1, args.epoch // 20)
    
    # ============================================= epoch loop begins =============================================
    # build wandb logger
    if dist.is_master():
        wandb_utils.wandb.init(project=args.project_name, name=args.exp_name, config={})
    for ep in range(args.epoch):
        # build data at each epoch to ensure read meta take effects for each dataloader worker
        args.epoch = ep

        if ep == 0:
            train_dataset = build_dataset(args)
            iters_train = len(train_dataset)
            start_ep = start_global_it // iters_train
            start_it = start_global_it % iters_train
            print(f'[PT info]  from ep{start_ep} it{start_it} {iters_train=}=======>  bed: {args.bed}  <=======\n')

        if ep < start_ep:
            continue
        if ep > start_ep:
            train_dataset = build_dataset(args)
            iters_train = len(train_dataset)

        # [train one epoch]
        train_dataloader = DataLoader(dataset=train_dataset, num_workers=args.workers, pin_memory=True, batch_size=None)
        stats = train_one_epoch(
            epoch=ep,
            is_first_ep=ep == start_ep,
            start_it=start_it if ep == start_ep else 0,
            start_global_it=start_global_it,
            me=None,
            saver=saver,
            args=args,
            dataloader_iter=iter(train_dataloader),
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
        )
        
        del stats, train_dataset, train_dataloader
    return


g_speed_ls = deque(maxlen=128)
def train_one_epoch(
    epoch: int, is_first_ep: bool, start_it: int, start_global_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args: arg_util.Args, dataloader_iter, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer,
):
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    step_cnt = 0
    header = f'[Ep]: [{epoch:4d}/{args.epoch}]'
    
    last_touch = time.time()
    g_it, max_it = epoch * iters_train, args.epoch * iters_train
    
    doing_profiling = args.prof and epoch == 0 and (args.profall or dist.is_master())
    maybe_record_function = record_function if doing_profiling else nullcontext
    trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function
    
    last_t_perf = time.time()
    speed_ls: deque = g_speed_ls
    FREQ = min(args.prof_freq, iters_train//2-1)
    NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
    ranges = set([2 ** i for i in range(20)])
    if epoch <= 1: ranges |= {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40}
    PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in ranges)

    me = misc.MetricLogger()
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['L', 'L_i', 'L_v']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Acc', 'Acc_i', 'Acc_v']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['seq_usage']]
    # ============================================= iteration loop begins =============================================
    for it, data in me.log_every(start_it, iters_train, dataloader_iter, args.log_freq, args.log_every_iter, header, args):
        g_it = epoch * iters_train + it
        # mfutool.step()
        # mfu_val = mfutool.get_mfu() * 100 # to percent
        # print(f"[MFU] step={g_it}, mfu={mfu_val:.2f} %, mfu.iter_time = {mfutool.iter_time():.4f} s")


        if (it+1) % FREQ == 0:
            speed_ls.append((time.time() - last_t_perf) / FREQ)
            last_t_perf = time.time()

        if (g_it+1) % args.save_model_iters_freq == 0:
            if args.checkpoint_type == 'torch':
                saver.sav(args=args, g_it=(g_it+1), next_ep=epoch, next_it=it+1, trainer=trainer, acc_str=f'[todo]', eval_milestone=None, also_save_to=None, best_save_to=None)
            elif args.checkpoint_type == 'omnistore':
                saver.sav(args=args, global_it=(g_it+1), next_ep=epoch, next_it=it+1, fsdp_object=trainer.gpt, optimizer_object=trainer.gpt_opt.optimizer, acc_str=None, eval_milestone=None)
        
        with maybe_record_function('before_train'):
            # [get data]
            images, captions, raw_features_bcthw, feature_cache_files4images, media = data['images'], data['captions'], data['raw_features_bcthw'], data['feature_cache_files4images'], data['media']

            # # [prepare text features]
            if args.text_tokenizer_type == 'flan_t5':
                tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
                input_ids = tokens.input_ids.cuda(non_blocking=True)
                mask = tokens.attention_mask.cuda(non_blocking=True)
                text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                lens: List[int] = mask.sum(dim=-1).tolist()
                cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                Ltext = max(lens)
                kv_compact = []
                for text_ind, (len_i, feat_i) in enumerate(zip(lens, text_features.unbind(0))):
                    kv_compact.append(feat_i[:len_i])
                kv_compact = torch.cat(kv_compact, dim=0)
                text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext)
            else:
                text_features = text_encoder(captions, args.device)
                lens = [len(item) for item in text_features]
                cu_seqlens_k = [0]
                for len_i in lens:
                    cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
                cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
                Ltext = max(lens)
                kv_compact = torch.cat(text_features, dim=0).float()
                text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)

            if len(images):
                images = [item.to(args.device, non_blocking=True) for item in images]
            if len(raw_features_bcthw):
                raw_features_bcthw = [item.to(args.device, non_blocking=True) for item in raw_features_bcthw]
            
            # [logging]
            if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                args.dump_log()
                last_touch = time.time()
                        
            # [get scheduled hyperparameters]
            progress = g_it / (max_it - 1)
            clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1
            
            stepping = (g_it + 1) % args.ac == 0
            step_cnt += int(stepping)
        
        with maybe_record_function('in_training'):
            grad_norm_t, scale_log2_t = trainer.train_step(
                epoch=epoch, 
                it=it, 
                g_it=g_it, 
                stepping=stepping, 
                clip_decay_ratio=clip_decay_ratio,
                metric_lg=me, 
                inp_B3HW=images, 
                raw_features_bcthw=raw_features_bcthw,
                feature_cache_files4images=feature_cache_files4images,
                text_cond_tuple=text_cond_tuple,
                media=media,
                args=args,
            )
        
        with maybe_record_function('after_train'):
            me.update(tlr=args.tlr)
    # ============================================= iteration loop ends =============================================
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


def main():    
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    main_train(args)
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    dist.barrier()

if __name__ == '__main__':
    main()
