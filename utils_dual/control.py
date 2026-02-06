import gradio as gr
import numpy as np
import torch
import requests 
import random
import os
import pdb
import sys
import copy
import json
import math
import types
import pickle
from PIL import Image
import base64
from io import BytesIO

from tqdm.auto import tqdm
from datetime import datetime
from safetensors.torch import load_file
from typing import List, Optional
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn

import diffusers
# from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import Attention
from torchvision.utils import save_image
from diffusers import DDIMScheduler, DiffusionPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
# from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
# from diffuser.pipeline_stable_diffusion_xl_dense import StableDiffusionXLDensePipeline
from transformers import CLIPTextModel, CLIPTokenizer

from .convert import convert_white_to_black
from .utils import *

Layers_control_net = {
                  "down_blocks":{
                                # "1":{
                                #     "0": list(range(0, 2)), 
                                #     "1": list(range(0, 2))
                                #     }, 
                                 "2":{
                                    "0": list(range(0, 10)), 
                                    "1": list(range(0, 10))
                                    }
                                 },
                  "mid_block":{"0":{"0": list(range(0, 10))}}, 
                }
Layers_dense = {
                  "down_blocks":{
                                # "1":{
                                #     "0": list(range(0, 2)), 
                                #     "1": list(range(0, 2))
                                #     }, 
                #                  "2":{
                #                     "0": list(range(0, 10)), 
                #                     "1": list(range(0, 10))
                #                     }
                #                  },
                #   "mid_block":{"0":{"0": list(range(0, 10))}}, 
                #   "up_blocks":{
                #                 "0":{
                #                     "0": list(range(0, 10)), 
                #                     "1": list(range(0, 10)), 
                #                      "2": list(range(0, 10))
                #                      }, 
                            #    "1":{
                            #        "0": list(range(0, 2)), 
                            #        "1": list(range(0, 2)), 
                            #        "2": list(range(0, 2))
                            #        }
                                }
                }

Layers_ALL = {
                  "down_blocks":{
                                "1":{
                                    "0": list(range(0, 2)), 
                                    "1": list(range(0, 2))
                                    }, 
                                 "2":{
                                    "0": list(range(0, 10)), 
                                    "1": list(range(0, 10))
                                    }
                                 },
                  "mid_block":{"0":{"0": list(range(0, 10))}}, 
                  "up_blocks":{
                                "0":{
                                    "0": list(range(0, 10)), 
                                    "1": list(range(0, 10)), 
                                     "2": list(range(0, 10))
                                     }, 
                               "1":{
                                   "0": list(range(0, 2)), 
                                   "1": list(range(0, 2)), 
                                   "2": list(range(0, 2))
                                   }
                                }
                }



# 使用单一循环处理所有条目
def process_layers_enabled(Layers_enabled):
    Layers_enabled_list = []
    for high_block, blocks in Layers_enabled.items():
        for block_key, block_two in blocks.items():
            for block_key2, block_list in block_two.items():
                for attention_index in block_list:
                    if high_block=="mid_block":
                        path = f"{high_block}.attentions.{block_key2}.transformer_blocks.{attention_index}"
                    else:
                        path = f"{high_block}.{block_key}.attentions.{block_key2}.transformer_blocks.{attention_index}"
                    Layers_enabled_list.append(path)
    return Layers_enabled_list

Layers_Enabled_ALL = process_layers_enabled(Layers_ALL)
Layers_Enabled_Control = process_layers_enabled(Layers_control_net)
Layers_Enabled_Dense = process_layers_enabled(Layers_dense)

def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (
        1,
        num_channels_latents,
        int(height) // self.vae_scale_factor,
        int(width) // self.vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents.repeat(batch_size, 1, 1, 1)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * self.scheduler.init_noise_sigma
    # pdb.set_trace()
    return latents

def drop_elements(features, embedding_drop, threshold=0.5, scale=None, step=0, switch="conv"):
    # features --> torch.Size([80, 1024, 64]), embedding_drop --> torch.Size([20, 4, 64])
    feat_dual = features[features.shape[0]//4*3:]
    feat_one = features[features.shape[0]//4*2:features.shape[0]//4*3]
    hw_shape = int(feat_dual.shape[1]**0.5)
    attn_drop = torch.baddbmm(torch.empty(feat_dual.shape[0], feat_dual.shape[1], embedding_drop.shape[1], 
                                        dtype=feat_dual.dtype, device=feat_dual.device),
                            feat_dual, embedding_drop.transpose(-1, -2), beta=0, alpha=scale)
    attn_drop_one = torch.baddbmm(torch.empty(feat_dual.shape[0], feat_dual.shape[1], embedding_drop.shape[1], 
                                        dtype=feat_dual.dtype, device=feat_dual.device),
                            feat_dual, embedding_drop.transpose(-1, -2), beta=0, alpha=scale)
    attn_drop = attn_drop.softmax(dim=-1)
    attn_drop_one = attn_drop_one.softmax(dim=-1)

    if False:
        reshaped_sample = attn_drop[0].T.reshape(embedding_drop.shape[1], int(feat_dual.shape[1]**0.5), int(feat_dual.shape[1]**0.5))
        for i in range(embedding_drop.shape[1]):
            plt.figure(figsize=(6, 6))  # 设置画布大小
            plt.imshow(reshaped_sample[i].cpu().numpy(), cmap='viridis')  # 使用 Viridis 色彩方案
            plt.colorbar(label="Value")  # 添加色条
            plt.title(f"Heatmap of Channel {i}")  # 设置标题
            plt.axis("off")  # 隐藏坐标轴
            plt.savefig(f"visual/drop_tree/{step}heatmap_channel_{i}.png")  # 保存图片
            plt.close()  # 关闭当前画布以避免重叠

    # pdb.set_trace()
    # 获取 attn 第一个索引的值，并基于 threshold 计算 mask
    attn_drop_one_idx = attn_drop_one[:, :, 0]  # Shape: [20, 1024]
    mask_none_empty = (attn_drop_one_idx > threshold).to(feat_dual.dtype)  # Shape: [20, 32, 32]
    attn_two_idx = attn_drop[:, :, 0]  # Shape: [20, 1024]
    mask = (attn_two_idx > threshold).to(feat_dual.dtype)  # Shape: [20, 32, 32]

    if switch=="zero":
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, feat_dual.shape[-1])  # Shape: [20, 1024, 64]
        feat_dual[expanded_mask.bool()] = 0
        features[features.shape[0] // 4 * 3:] = feat_dual
        return features

    mask = mask.view(attn_two_idx.shape[0], hw_shape, hw_shape)  # Shape: [20, 32, 32]
    # 将 feat_dual 重塑为 32x32 空间
    feat_dual_reshaped = feat_dual.view(feat_dual.shape[0], hw_shape, hw_shape, feat_dual.shape[-1])  # Shape: [20, 32, 32, 64]

    # 定义 5x5 卷积核
    kernel = torch.ones((1, 1, 5, 5), dtype=feat_dual.dtype, device=feat_dual.device)
    kernel[0, 0, 2, 2] = 0  # 忽略中心点
    kernel = kernel.expand(feat_dual_reshaped.shape[-1], 1, 5, 5)  # 扩展到所有通道  

    # 计算周围点的加权和，但排除 mask=1 的点
    inverted_mask = 1 - mask  # 反转 mask
    masked_feat_dual = feat_dual_reshaped.permute(0, 3, 1, 2) * inverted_mask.unsqueeze(1)  # 保留 mask=0 的点
    
    # 计算邻居权重总和
    masked_feat_dual = masked_feat_dual.contiguous()  # 转换为 [batch_size, channels, height, width]
    neighbor_sum = F.conv2d(masked_feat_dual, kernel, padding=2, groups=masked_feat_dual.shape[1])  # Shape: [10, 64, 64, 64]

    # 计算邻居权重总和
    neighbor_weights = F.conv2d(inverted_mask.unsqueeze(1), kernel, padding=2)  # Shape: [10, 1, 64, 64]
    # pdb.set_trace()q

    # 避免除以零，计算有效邻居的均值
    neighbor_mean = torch.where(
        neighbor_weights > 0,
        neighbor_sum / neighbor_weights,
        torch.full_like(neighbor_sum, 0.)  # 没有有效邻居时用 default_value
    )

    # 将大于阈值的点替换为周围点的均值
    updated_feat_dual = feat_dual_reshaped.permute(0, 3, 1, 2) * (1 - mask.unsqueeze(1)) + neighbor_mean * mask.unsqueeze(1)

    # 恢复更新后的 feat_dual
    updated_feat_dual = updated_feat_dual.permute(0, 2, 3, 1).view_as(feat_dual)

    # 替换 features 中的 feat_dual 部分
    features[features.shape[0] // 4 * 3:] = updated_feat_dual
    # features[features.shape[0]//4*2:features.shape[0]//4*3] = updated_feat_dual

    return features, mask_none_empty

def blance_text_embeddings(cond_embeddings_first, DECODED_PROMPTS, high_noun_indices, beta=1., beta_color=0.75, beta_adj=1, beta_det=1, switch=2):
    mask_embeddings = torch.ones((77,1)).to(cond_embeddings_first.device)
    energy = torch.sum(cond_embeddings_first**2, axis=-1, keepdim=False).squeeze(0)
    # switch = 1 # 3 is good to explain
    if switch==1:
        cof = torch.sqrt(beta*energy[len(DECODED_PROMPTS)]/energy[high_noun_indices].max())
        for idx in high_noun_indices:
            mask_embeddings[idx] = cof
    elif switch==2: 
        for idx in high_noun_indices:
            cof = torch.sqrt(beta*energy[len(DECODED_PROMPTS)]/energy[idx])
            # print(energy[len(DECODED_PROMPTS)], energy[idx], cof)
            mask_embeddings[idx] = cof
    else:
        mask_embeddings[high_noun_indices] *= torch.sqrt(beta*energy[high_noun_indices].max()/energy[high_noun_indices]).unsqueeze(-1)

        # for idx in high_noun_indices:
        #     cof = torch.sqrt(beta*energy[high_noun_indices].max()/energy[idx])
        #     # print(energy[len(DECODED_PROMPTS)], energy[idx], cof)
        #     mask_embeddings[idx] = cof

    text = " ".join(DECODED_PROMPTS[1:])
    doc = nlp(text)

    color_indices = [i + 1 for i, token in enumerate(doc) if is_color_word(token.text.lower())]
    adjective_indices = [i+1 for i, token in enumerate(doc) if token.pos_ == 'ADJ']
    determiner_indices = [i+1 for i, token in enumerate(doc) if token.pos_ == 'DET'] # "a, the, my"
    # pdb.set_trace()
    if beta_color!=1:
        # mask_embeddings[color_indices] *= torch.sqrt(beta_color*energy[high_noun_indices].max()/energy[color_indices]).unsqueeze(-1)
        mask_embeddings[adjective_indices] *= torch.sqrt(beta_adj*energy[high_noun_indices].max()/energy[adjective_indices]).unsqueeze(-1)

    cond_embeddings_first = mask_embeddings*cond_embeddings_first

    return cond_embeddings_first


def amplify_max_min(tensor, alpha=2.):
    max_vals, _ = torch.max(tensor, dim=-2, keepdim=True)
    min_vals, _ = torch.min(tensor, dim=-2, keepdim=True) 
    # 对最大值的处理
    amplified_tensor = torch.where(
        (tensor == max_vals) & (max_vals > 0), tensor * alpha,  # 放大正的最大值
        torch.where((tensor == max_vals) & (max_vals < 0), tensor / alpha, tensor)  # 缩小负的最大值
    )

    # 对最小值的处理
    amplified_tensor = torch.where(
        (tensor == min_vals) & (min_vals > 0), amplified_tensor / alpha,  # 缩小正的最小值
        torch.where((tensor == min_vals) & (min_vals < 0), amplified_tensor * alpha, amplified_tensor)  # 放大负的最小值
    )
    return amplified_tensor


def amplify_feature_topk(hidden_states, value, mask, N=1, alpha=1, high_noun_indices=None, max_token=None, is_dual=False):
    if N==0: return hidden_states

    # if len(hidden_states.shape)==4: pdb.set_trace()
    # if is_dual: mask[..., high_noun_indices[1]] = (0.75*torch.ones(mask[..., high_noun_indices[1]].shape)).to(mask.dtype)
    mask_expanded = mask.unsqueeze(-1).repeat_interleave(value.shape[-1], dim=-1)
    if len(hidden_states.shape)==3:
        high_noun_indices = torch.tensor(high_noun_indices).to(value.device).unsqueeze(-1).unsqueeze(0)
        high_noun_indices = high_noun_indices.repeat(value.shape[0], 1, value.shape[-1])
    else:
        high_noun_indices = torch.tensor(high_noun_indices).to(value.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        high_noun_indices = high_noun_indices.repeat(value.shape[0], value.shape[1], 1, value.shape[-1])
    topk_max_values, topk_max_indices = torch.topk(torch.abs(value[..., 1:max_token, :] if max_token is not None else value), N, dim=-2)
    topk_max_indices = topk_max_indices + 1
    value_mask = torch.zeros_like(value)
    value_mask_high_noun = torch.zeros_like(value)
    value_mask.scatter_(-2, topk_max_indices, 1)
    value_mask_high_noun.scatter_(-2, high_noun_indices, 1)
    value_mask_combine = value_mask * value_mask_high_noun
    hidden_states_mask = torch.sum(value_mask_combine.unsqueeze(-3) * mask_expanded, dim=-2)

    if is_dual:
        hidden_states[int(hidden_states.size(0)//4*3):] = (hidden_states_mask>0)*alpha*hidden_states[int(hidden_states.size(0)//4*3)]
    else:
        hidden_states[int(hidden_states.size(0)/2):int(hidden_states.size(0)//4*3)] = (hidden_states_mask>0)*alpha*hidden_states[int(hidden_states.size(0)/2):int(hidden_states.size(0)//4*3)]
    return hidden_states


def amplify_top_n_max_min(tensor, N=1, alpha=1., mask=None, high_noun_indices=None, max_token=None):
    if N == 0:
        return tensor

    if mask is not None:
        tensor = tensor * (mask.unsqueeze(0).to(tensor.dtype))

    # 排序张量以找到最大值
    sorted_tensor, sorted_indices = torch.sort(torch.abs(tensor[..., 1:max_token, :]), dim=-2) if max_token is not None else torch.sort(tensor, dim=-2)
    sorted_indices = sorted_indices + 1

    fn_type = "amplify_max"  # 改为只处理极大值
    if fn_type != "amplify_max":
        high_noun_indices = None 

    if high_noun_indices is not None:
        noun_mask = torch.zeros(tensor.size(-2), dtype=torch.bool, device=tensor.device)
        noun_mask[high_noun_indices] = True
    else:
        noun_mask = torch.ones(tensor.size(-2), dtype=torch.bool, device=tensor.device)

    if fn_type == "amplify_max":
        amplified_tensor = tensor.clone()

        # 只处理极大值
        for i in range(1, N + 1):
            current_max_vals = sorted_tensor[..., -i, :].unsqueeze(-2)  # 倒数第 i 大的值
            current_max_indices = sorted_indices[..., -i, :].unsqueeze(-2) 
            valid_max_mask = noun_mask[current_max_indices]

            amplified_tensor = torch.where(
                (tensor == current_max_vals) & valid_max_mask, 
                amplified_tensor * alpha,  # 放大正的极大值
                amplified_tensor
            )

    return amplified_tensor

def amplify_top_n_max_min_AA(tensor, N=1, alpha=1., mask=None, high_noun_indices=None, max_token=None):
    if N==0: return tensor
    if mask is not None:
        tensor = tensor * (mask.unsqueeze(0).to(tensor.dtype))

    sorted_tensor, sorted_indices = torch.sort(tensor[..., 1:max_token, :], dim=-2) if max_token is not None else torch.sort(tensor, dim=-2)
    # sorted_tensor, sorted_indices = torch.sort(tensor, dim=-2)
    # pdb.set_trace()
    sorted_indices = sorted_indices + 1

    fn_type = "amplify_max_min" # "zero_others", "zero_max_min", "amplify_max_min"
    if fn_type != "amplify_max_min": high_noun_indices = None 

    if high_noun_indices is not None:
        noun_mask = torch.zeros(tensor.size(-2), dtype=torch.bool, device=tensor.device)
        noun_mask[high_noun_indices] = True
    else:
        noun_mask = torch.ones(tensor.size(-2), dtype=torch.bool, device=tensor.device)

    if fn_type == "zero_others":
        total_mask = torch.cat((
                                # torch.ones_like(tensor[..., 0:1, :], dtype=torch.bool),
                               torch.zeros_like(tensor[..., 0:max_token, :], dtype=torch.bool),
                            torch.ones_like(tensor[..., max_token:, :], dtype=torch.bool)),
                            dim=-2)

        # pdb.set_trace()
        # 处理极大值
        for i in range(1, N + 1):
            current_max_vals = sorted_tensor[..., -i, :].unsqueeze(-2)
            current_max_indices = sorted_indices[..., -i, :].unsqueeze(-2)
            valid_max_mask = noun_mask[current_max_indices]

            current_max_mask = torch.isclose(tensor, current_max_vals) & valid_max_mask
            total_mask |= current_max_mask

        # 处理极小值
        for i in range(N):
            current_min_vals = sorted_tensor[..., i, :].unsqueeze(-2)
            current_min_indices = sorted_indices[..., i, :].unsqueeze(-2)
            valid_min_mask = noun_mask[current_min_indices]

            current_min_mask = torch.isclose(tensor, current_min_vals) & valid_min_mask
            total_mask |= current_min_mask

        # 将非极值位置置零
        amplified_tensor = tensor * total_mask.to(tensor.dtype)

    elif fn_type == "zero_max_min":
        amplified_tensor = tensor.clone()
        for i in range(1, N + 1):
            current_max_indices = sorted_indices[..., -i, :]
            # 将这些位置的值置零
            amplified_tensor.scatter_(-2, current_max_indices.unsqueeze(-2), torch.zeros_like(current_max_indices.unsqueeze(-2).to(tensor.dtype)))

        # 处理极小值：找到 N 个极小值并将它们置零
        for i in range(N):
            current_min_indices = sorted_indices[..., i, :]
            # 将这些位置的值置零
            amplified_tensor.scatter_(-2, current_min_indices.unsqueeze(-2), torch.zeros_like(current_min_indices.unsqueeze(-2).to(tensor.dtype)))
    
    elif fn_type == "amplify_max_min":
        amplified_tensor = tensor.clone()
        # pdb.set_trace()
        for i in range(1, N + 1):
            current_max_vals = sorted_tensor[..., -i, :].unsqueeze(-2)  # 倒数第 i 大的值
            current_max_indices = sorted_indices[..., -i, :].unsqueeze(-2) 
            valid_max_mask = noun_mask[current_max_indices]

            amplified_tensor = torch.where(
                (tensor == current_max_vals) & (current_max_vals > 0) & valid_max_mask, 
                amplified_tensor * alpha,  # 放大正的极大值
                torch.where((tensor == current_max_vals) & (current_max_vals < 0) & valid_max_mask, 
                            amplified_tensor / alpha, amplified_tensor)  # 缩小负的极大值
            )
        
        for i in range(N):
            current_min_vals = sorted_tensor[..., i, :].unsqueeze(-2)  # 正数第 i 小的值
            current_min_indices = sorted_indices[..., i, :].unsqueeze(-2) 
            valid_min_mask = noun_mask[current_min_indices]

            amplified_tensor = torch.where(
                (tensor == current_min_vals) & (current_min_vals > 0) & valid_min_mask, 
                amplified_tensor / alpha,  # 缩小正的极小值
                torch.where((tensor == current_min_vals) & (current_min_vals < 0) & valid_max_mask, 
                            amplified_tensor * alpha, amplified_tensor)  # 放大负的极小值
            )
    #     try:
    #         print(amplified_tensor[0, 0, :, 0])
    #     except:
    #         print(amplified_tensor[0, :, 0])

    # return torch.zeros_like(tensor)
    return amplified_tensor


def init_latent_with_boxes(in_channels, height, width, batch_size, layouts, seed):
    # pdb.set_trace()
    # Step 1: 初始化背景latent0
    init_latent = torch.zeros((batch_size, in_channels, height, width))
    generator = torch.Generator().manual_seed(seed)
    # init_latent = torch.randn(
    #     (batch_size, in_channels, height, width),
    #     generator=generator,
    # )
    # Step 2: 对每个box生成新的latent片段并替换对应位置
    for i, layout in enumerate(layouts):
        # 为每个segment随机选择一个种子
        segment_seed = random.randint(0, 1000000000)
        generator.manual_seed(segment_seed)
        
        # 创建segment mask
        mask = layout.unsqueeze(0).repeat(batch_size, in_channels, 1, 1)
        
        # 生成segment对应的latent片段
        segment_latent = torch.randn((batch_size, in_channels, height, width), generator=generator) * (1.0 + 0.1*i)
        
        # 将segment latent叠加到背景latent
        init_latent = init_latent + segment_latent * mask.cpu()
    
        # pdb.set_trace()
    # save_image(init_latent[0, 0], "init_latent.png")
    init_latent /= init_latent.std()
        
    return init_latent

class CustomSaveImagePipeline(StableDiffusionXLControlNetPipeline):
    def __call__(self, *args, **kwargs):
        save_steps = kwargs.pop('save_steps', [0, 1, 2, 3, 7, 15, 23, 31])  # Save image every 'save_steps'
        callback = kwargs.pop('callback', None)
        save_dir = kwargs.pop('save_dir', "visual/temp")
        os.makedirs(save_dir, exist_ok=True)   
        
        def save_image_callback(step: int, timestep: int, latents: torch.FloatTensor):
            if step in save_steps:  # Save image at specific intervals
                with torch.no_grad():
                    self.upcast_vae()
                    latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                    latents = latents / self.vae.config.scaling_factor
                    image = self.vae.decode(latents, return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type="pil")
                    for ii, oo in enumerate(image):
                        oo.save(f"{save_dir}/step_{step}_{ii}.png")
                    self.vae.to(dtype=torch.float16)
            if callback:
                callback(step, timestep, latents)

        kwargs['callback'] = save_image_callback
        kwargs['callback_steps'] = 1  # Call callback every step

        return super().__call__(*args, **kwargs)


def save_attn_map(sim, sub_path, DECODED_PROMPTS, sa_, PROMPTS_LIST, postfix=""):
    os.makedirs(sub_path, exist_ok=True)
    if len(sim.size())==4:
        sim = sim[2]
        idx_fix = 1
    else:
        idx_fix = int(sim.shape[0]//2)
    hh = int(np.sqrt(sim.shape[1]))
    if sa_:
        save_imshow(sim[idx_fix,:,:], 
                    f"{sub_path}/self_iso_all{postfix}.png")
        for idx in [34, 175, 512, 856]:    
            save_imshow(sim[idx_fix,idx,:].view(hh, hh), 
                    f"{sub_path}/self_iso_postion_{idx}{postfix}.png")
    else:
        for idx in PROMPTS_LIST:    
            if idx<len(DECODED_PROMPTS):
                save_imshow(sim[idx_fix,:,idx].view(hh, hh), 
                            f"{sub_path}/cross_iso_{DECODED_PROMPTS[idx]}{postfix}.png",
                            # scale2=sim[idx_fix,:,:].max()
                            )

def save_value_map(value, sub_path, num=20, postfix=""):
    os.makedirs(sub_path, exist_ok=True)
    if len(value.size())==4:
        value = value[2]
        idx_fix = 1
    else:
        idx_fix = int(value.shape[0]//2)
        # for idx in [40, 41, 42, 44]:
    for idx in [idx_fix]:
        save_imshow(value[idx,0:40,:], 
                f"{sub_path}/value{idx}{postfix}.png")
        save_plot(value[idx,0:40,:], f"{sub_path}/value_plot_{idx}{postfix}.png", line_num=num, lines_per_subplot=1)
    np.save(f"{sub_path}/value{postfix}.npy", value.cpu().numpy())

def save_hidden_map(hidden_states, sub_path, num=20, postfix=""):
    os.makedirs(sub_path, exist_ok=True)
    if len(hidden_states.size())==4:
        hidden_states = hidden_states[2]
        idx_fix = 1
    else:
        idx_fix = int(hidden_states.shape[0]//2)
    for idx in range(num):    
        hh = int(np.sqrt(hidden_states.shape[1]))
        save_imshow(hidden_states[idx_fix,:,idx].view(hh, hh), 
                    f"{sub_path}/hidden_ch{idx}.png",
                    # scale2=hidden_states[idx_fix,:,:num].max()
                    )
        
# def mod_forward_ip(self,
#         # attn: Attention,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         temb: Optional[torch.Tensor] = None,
#         scale: float = 1.0,
#         ip_adapter_masks: Optional[torch.Tensor] = None,
#     ):
#     residual = hidden_states
#     # print(f"ip:{self}")
#     # separate ip_hidden_states from encoder_hidden_states
#     # pdb.set_trace()
#     if encoder_hidden_states is not None:
#         if isinstance(encoder_hidden_states, tuple):
#             encoder_hidden_states, ip_hidden_states = encoder_hidden_states
#         else:
#             deprecation_message = (
#                 "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
#                 " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
#             )
#             deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
#             end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
#             encoder_hidden_states, ip_hidden_states = (
#                 encoder_hidden_states[:, :end_pos, :],
#                 [encoder_hidden_states[:, end_pos:, :]],
#             )

#     if self.spatial_norm is not None:
#         hidden_states = self.spatial_norm(hidden_states, temb)

#     input_ndim = hidden_states.ndim

#     if input_ndim == 4:
#         batch_size, channel, height, width = hidden_states.shape
#         hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#     batch_size, sequence_length, _ = (
#         hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#     )
#     if attention_mask is not None:
#         attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#         # scaled_dot_product_attention expects attention_mask shape to be
#         # (batch, heads, source_length, target_length)
#         attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

#     if self.group_norm is not None:
#         hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#     query = self.to_q(hidden_states)

#     global sreg, creg, COUNT, COUNT_DUAL, creg_maps, sreg_maps, reg_sizes, text_cond, dense_step, sep_step
#     if DEBUG_ATTN and (int(COUNT/ALL_LAYERS) in DEBUG_STEP): global cross_maps_dict1, cross_maps_dict2, cross_feat_dict
    
#     # pdb.set_trace()
#     sa_ = True if encoder_hidden_states is None else False
#     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
#     # if COUNT/ALL_LAYERS <= sep_step:
#     #     # print(f"sep:{COUNT/ALL_LAYERS}") 
#     #     encoder_hidden_states = text_cond if encoder_hidden_states is not None else hidden_states
#     # else:
#     #     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
#     if self.norm_cross:
#         encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

#     key = self.to_k(encoder_hidden_states)
#     value = self.to_v(encoder_hidden_states)

#     inner_dim = key.shape[-1]
#     head_dim = inner_dim // self.heads

#     query_ip = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#     query = self.head_to_batch_dim(query)
#     key = self.head_to_batch_dim(key)
#     value = self.head_to_batch_dim(value)
    
#     if COUNT/ALL_LAYERS < dense_step:
#         #  32 = (16 self + 16 cross), no more than 15 steps
#         dtype = query.dtype
#         if self.upcast_attention:
#             query = query.float()
#             key = key.float()
#         # Self-attn: query/key/value: torch.Size([16, 4096, 40])
#         # Cross-attn: query/key/value: torch.Size([16, 4096, 40])
#         sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
#                                         dtype=query.dtype, device=query.device),
#                             query, key.transpose(-1, -2), beta=0, alpha=self.scale)
#         # print(f"steps:{COUNT/ALL_LAYERS}, {self.scale}")
#         # out=β×input+α×(batch1×batch2), self.scale=0.15811388300841897, torch.Size([16, 4096, 4096])
#         # pdb.set_trace()
#         if DEBUG_ATTN and (int(COUNT/ALL_LAYERS) in DEBUG_STEP): cross_maps_dict1[int(COUNT/ALL_LAYERS)].append(
#             ((sim.view(-1, self.heads, query.shape[1], key.shape[1])).abs().sum(1).detach().cpu().numpy())[...,:32])

#         # treg = 1
#         treg = torch.pow(timesteps[COUNT//ALL_LAYERS]/1000, 1)
#         # the treg value is expoential attenuation as the the timestep from 1000 to 0.
#         # the modulation co-efficient of self-attention sreg=0.3, while the cross-attention creg=1

#         ## reg at cross-attn
#         if not sa_:
#             min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
#             max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
#             mask = creg_maps[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
#             size_reg = reg_sizes[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)

#             # if int(COUNT/ALL_LAYERS)==8: pdb.set_trace()
#             # save_image((sim[int(sim.size(0)/2):].abs().sum(0))[0].view(64, 64), "cross_sim.jpg")
#             sim[int(sim.size(0)/2):] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/2):])
#             sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/2):]-min_value)

#         if DEBUG_ATTN and (int(COUNT/ALL_LAYERS) in DEBUG_STEP): cross_maps_dict2[int(COUNT/ALL_LAYERS)].append(
#             ((sim.view(-1, self.heads, query.shape[1], key.shape[1])).abs().sum(1).detach().cpu().numpy())[...,:32])
#         attention_probs = sim.softmax(dim=-1)
#         attention_probs = attention_probs.to(dtype)
            
#     else:
#         attention_probs = self.get_attention_scores(query, key, attention_mask)
#         if DEBUG_ATTN and (int(COUNT/ALL_LAYERS) in DEBUG_STEP): cross_maps_dict1[int(COUNT/ALL_LAYERS)].append(
#             ((attention_probs.view(-1, self.heads, query.shape[1], key.shape[1])).abs().sum(1).detach().cpu().numpy())[...,:32])
        
#     COUNT += 1
#     COUNT_DUAL +=1
            
#     hidden_states = torch.bmm(attention_probs, value)
#     hidden_states = self.batch_to_head_dim(hidden_states)

#     # pdb.set_trace()
#     if ip_adapter_masks is not None:
#         if not isinstance(ip_adapter_masks, List):
#             # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
#             ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
#         if not (len(ip_adapter_masks) == len(self.processor.scale) == len(ip_hidden_states)):
#             raise ValueError(
#                 f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
#                 f"length of self.scale array ({len(self.processor.scale)}) and number of ip_hidden_states "
#                 f"({len(ip_hidden_states)})"
#             )
#         else:
#             for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.processor.scale, ip_hidden_states)):
#                 if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
#                     raise ValueError(
#                         "Each element of the ip_adapter_masks array should be a tensor with shape "
#                         "[1, num_images_for_ip_adapter, height, width]."
#                         " Please use `IPAdapterMaskProcessor` to preprocess your mask"
#                     )
#                 if mask.shape[1] != ip_state.shape[1]:
#                     raise ValueError(
#                         f"Number of masks ({mask.shape[1]}) does not match "
#                         f"number of ip images ({ip_state.shape[1]}) at index {index}"
#                     )
#                 if isinstance(scale, list) and not len(scale) == mask.shape[1]:
#                     raise ValueError(
#                         f"Number of masks ({mask.shape[1]}) does not match "
#                         f"number of scales ({len(scale)}) at index {index}"
#                     )
#     else:
#         ip_adapter_masks = [None] * len(self.processor.scale)

#     # for ip-adapter
#     for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
#         ip_hidden_states, self.processor.scale, self.processor.to_k_ip, self.processor.to_v_ip, ip_adapter_masks
#     ):
#         skip = False
#         if isinstance(scale, list):
#             if all(s == 0 for s in scale):
#                 skip = True
#         elif scale == 0:
#             skip = True
#         if not skip:
#             if mask is not None:
#                 if not isinstance(scale, list):
#                     scale = [scale] * mask.shape[1]

#                 current_num_images = mask.shape[1]
#                 for i in range(current_num_images):
#                     ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
#                     ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

#                     ip_key = ip_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#                     ip_value = ip_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

#                     # the output of sdp = (batch, num_heads, seq_len, head_dim)
#                     # TODO: add support for attn.scale when we move to Torch 2.1
#                     _current_ip_hidden_states = F.scaled_dot_product_attention(
#                         query_ip, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
#                     )

#                     _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
#                         batch_size, -1, self.heads * head_dim
#                     )
#                     _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

#                     mask_downsample = IPAdapterMaskProcessor.downsample(
#                         mask[:, i, :, :],
#                         batch_size,
#                         _current_ip_hidden_states.shape[1],
#                         _current_ip_hidden_states.shape[2],
#                     )

#                     mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
#                     hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
#             else:
#                 ip_key = to_k_ip(current_ip_hidden_states)
#                 ip_value = to_v_ip(current_ip_hidden_states)

#                 ip_key = ip_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#                 ip_value = ip_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

#                 # the output of sdp = (batch, num_heads, seq_len, head_dim)
#                 # TODO: add support for attn.scale when we move to Torch 2.1
#                 # pdb.set_trace()
#                 current_ip_hidden_states = F.scaled_dot_product_attention(
#                     query_ip, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
#                 )

#                 current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
#                     batch_size, -1, self.heads * head_dim
#                 )
#                 current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

#                 hidden_states = hidden_states + scale * current_ip_hidden_states

#     # linear proj
#     hidden_states = self.to_out[0](hidden_states)
#     # dropout
#     hidden_states = self.to_out[1](hidden_states)

#     if input_ndim == 4:
#         hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#     if self.residual_connection:
#         hidden_states = hidden_states + residual

#     hidden_states = hidden_states / self.rescale_output_factor
#     if DEBUG_ATTN and (int(COUNT/ALL_LAYERS) in DEBUG_STEP): 
#         cross_feat_dict[int(COUNT/ALL_LAYERS)].append(hidden_states[20:30].abs().sum(-1).detach().cpu().numpy())

#     return hidden_states

# def mod_forward_sd(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
#     residual = hidden_states
#     # print(hidden_states.shape)

#     if self.spatial_norm is not None:
#         hidden_states = self.spatial_norm(hidden_states, temb)

#     input_ndim = hidden_states.ndim

#     if input_ndim == 4:
#         batch_size, channel, height, width = hidden_states.shape
#         hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#     # print(self)
#     # if type(encoder_hidden_states)==tuple: pdb.set_trace()
#     batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
#     attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

#     if self.group_norm is not None:
#         hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#     query = self.to_q(hidden_states)

#     global sreg, creg, COUNT, COUNT_DUAL, creg_maps, sreg_maps, dual_maps, reg_sizes, text_cond, dense_step, sep_step, alpha
#     if DEBUG_ATTN and (int(COUNT/DENSE_LAYERS) in DEBUG_STEP): global self_maps_dict1, self_maps_dict2, self_feat_dict1
    
#     sa_ = True if encoder_hidden_states is None else False
#     encoder_hidden_states = text_cond if encoder_hidden_states is not None else hidden_states
#     # if COUNT/DENSE_LAYERS <= sep_step:
#     #     # print(f"sep:{COUNT/DENSE_LAYERS}") 
#     #     encoder_hidden_states = text_cond if encoder_hidden_states is not None else hidden_states
#     # else:
#     #     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
#     if self.norm_cross:
#         encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

#     key = self.to_k(encoder_hidden_states)
#     value = self.to_v(encoder_hidden_states)

#     query = self.head_to_batch_dim(query)
#     if sa_:
#         key = self.head_to_batch_dim(key)
#         value = self.head_to_batch_dim(value)
#     else:
#         key = self.head_to_batch_dim(key)
#         # key[key.shape[0]//2:] *= PROMPT_MASK.unsqueeze(0).to(key.dtype)
#         value = self.head_to_batch_dim(value)

#     if COUNT/DENSE_LAYERS < dense_step:
#         #  32 = (16 self + 16 cross), no more than 15 steps
#         dtype = query.dtype
#         if self.upcast_attention:
#             query = query.float()
#             key = key.float()
#         sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
#                                         dtype=query.dtype, device=query.device),
#                             query, key.transpose(-1, -2), beta=0, alpha=self.scale)
#         # out=β×input+α×(batch1×batch2), self.scale=0.15811388300841897, torch.Size([16, 4096, 4096])
        
#         # treg = 1
#         treg = torch.pow(timesteps[COUNT_DUAL//ALL_LAYERS]/1000, 5)

#         if DEBUG_ATTN and (int(COUNT_DUAL/ALL_LAYERS) in DEBUG_STEP) and (int(COUNT_DUAL%ALL_LAYERS) in DEBUG_Layer_Dense):
#             sub_path = f"{DEBUG_PATH}/sd_step{int(COUNT_DUAL//ALL_LAYERS)}_attn{int(COUNT_DUAL%ALL_LAYERS)}"
#             save_attn_map(sim, sub_path, DECODED_PROMPTS, sa_, [0]+HIGH_NOUN, postfix="_prior")

#         ## reg at self-attn
#         if sa_:
#             min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1) # torch.Size([8, 4096, 1])
#             max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1) # torch.Size([8, 4096, 1])
#             mask = sreg_maps[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
#             size_reg = reg_sizes[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
            
#             # sim[int(sim.size(0)/2):] += (mask>0)*size_reg*sreg*treg*(max_value-sim[int(sim.size(0)/2):])
#             # sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*sreg*treg*(sim[int(sim.size(0)/2):]-min_value)
            
#         ## reg at cross-attn
#         else:
#             min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
#             max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
#             mask = creg_maps[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
#             size_reg = reg_sizes[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)

#             # sim[int(sim.size(0)/2):] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/2):])
#             sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/2):]-min_value)

#         attention_probs = sim.softmax(dim=-1)
#         # if not sa_: 
#             # attention_probs = torch.where(attention_probs > 0.1, torch.tensor(1.0), attention_probs)
#         attention_probs = attention_probs.to(dtype)
            
#     else:
#         attention_probs = self.get_attention_scores(query, key, attention_mask)

#     if sa_:
#         hidden_states = torch.bmm(attention_probs, value)
#     else:
#         hidden_states = torch.bmm(attention_probs, amplify_top_n_max_min(value, N=BETA_NUM, alpha=BETA, high_noun_indices=HIGH_NOUN, max_token=len(DECODED_PROMPTS)))
#         mask = creg_maps[hidden_states.size(-2)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
#         hidden_states = amplify_feature_topk(hidden_states, value, mask, N=FEAT_NUM, alpha=FEAT_V, \
#                                              high_noun_indices=HIGH_NOUN, max_token=len(DECODED_PROMPTS)) 

#     hidden_states = self.batch_to_head_dim(hidden_states)

#     # linear proj
#     hidden_states = self.to_out[0](hidden_states)
#     # dropout
#     hidden_states = self.to_out[1](hidden_states)

#     if input_ndim == 4:
#         hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#     if self.residual_connection:
#         hidden_states = hidden_states + residual

#     hidden_states = hidden_states / self.rescale_output_factor

#     hidden_states = bridge_harmony_hidden(hidden_states, COUNT_DUAL, sep_step, alpha, dual_maps)
#     COUNT_DUAL += 1

#     return hidden_states  control 