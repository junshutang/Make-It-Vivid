import torch
import os
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from collections import defaultdict


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

# model_base = "/youtu_xuanyuan_shuzhiren_2906355_cq10/private/juliatang/sd_data/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
model_base = "/youtu_xuanyuan_shuzhiren_2906355_cq10/private/juliatang/sd_data/models--runwayml--stable-diffusion-v1-5/snapshots/stable-diffusion-v1-5"
# model_base = "/youtu_xuanyuan_shuzhiren_2906355_cq10/private/juliatang/sd_data/models--SG161222--Realistic_Vision_V1.4/snapshots/686d3dcb8bbc7e6a7757fd161e0fbafd23d6c629"
# model_base = "/youtu_xuanyuan_shuzhiren_2906355_cq10/private/juliatang/sd_data/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/c073febe0ec7ac20ab410dba7c90f10bce84caf6"
lora_model_path_list = ["lora/uv-bs4-r8"]
second_lora = "civi/more_details.safetensors"
third_lora = "lora_civi/MoXinV1.safetensors"
# prompt = "A 3D wearing suit, photo realistic."
# prompt_list = ["A monkey.", "A monkey wearing suit.", "A monkey wearing blue overall.", "A monkey wearing pink dress."]
# prompt_list = ["A fox.", "A fox wearing suit.", "A fox wearing blue overall.", "A fox wearing pink dress."]
added_prompt = '4k, detailed , digital art, trending arstation, film still "x", cinematic, intricate details, disney , pixar'
negative_prompt = 'low quality, stripe, word'
trigger = 'shuimo'
prompt_list = ["A rabbit wearing blue overalls."]

for lora_model_path in lora_model_path_list:
    pipeline = DiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, safety_checker=None)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(lora_model_path)
    # pipeline.unet.load_attn_procs(lora_model_path)
    pipeline = pipeline.to("cuda")
    
    # weight = 1.0
    # pipeline = load_lora_weights(pipeline, second_lora, weight, 'cuda', torch.float32)
    
    # weight = 0.7
    # pipeline = load_lora_weights(pipeline, third_lora, weight, 'cuda', torch.float32)


    for seed in range(10):
        torch.manual_seed(seed)
        for prompt in prompt_list:
            output_dir = "result/" + lora_model_path_list[0].split("/")[-1] + "_real"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{prompt.split('.')[0].replace(' ', '_')}_{seed}.png")
            # prompt = prompt_head + 
            prompt = prompt + ", " + added_prompt
            # prompt = prompt + ", " + added_prompt + ", " + trigger 
            
            image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

            image.save(save_path)