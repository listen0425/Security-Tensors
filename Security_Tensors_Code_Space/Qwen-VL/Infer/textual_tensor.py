import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import copy
import json
import random
import pandas as pd
import fire
import torch.optim as optim
import sys

sys.path.append("Qwen-VL-Chat") 
# Pass in the path of your Qwen-VL-Chat model

from visual import VisionTransformer
from qwen_generation_utils import make_context
def build_inputs_embeds(
    model,  # QWenModel
    input_ids: torch.LongTensor,
):
    # copy from modeling_qwen.py 
    if torch.any(input_ids == model.config.visual['image_start_id']):
        bos_pos = torch.where(input_ids == model.config.visual['image_start_id'])
        eos_pos = torch.where(input_ids == model.config.visual['image_start_id'] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        images = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1 : b - 1].tolist()
            image = image[ : image.index(model.config.visual['image_start_id'] + 2)]
            images.append(bytes(image).decode('utf-8'))

        images = model.visual.encode(images)
        assert images.shape[0] == len(images)
        fake_images = None
    elif model.training:
        fake_images=torch.zeros(1,3,224,224).to(
            dtype=model.visual.conv1.weight.dtype, device=model.visual.conv1.weight.device)
        images = model.visual(fake_images)
    else:
        fake_images = None
        images = None
    inputs_embeds = model.wte(input_ids)
    if fake_images is not None:
        inputs_embeds = inputs_embeds + images.mean()*0
    elif images is not None:
        for idx, (i, a, b) in enumerate(img_pos):
            inputs_embeds[i][a + 1 : b] = images[idx]
    return inputs_embeds


def infer(model, tokenizer, data,if_add_template, max_new_tokens=16):
    input_text = data["input"]
    image_path = data["pic_path"]
    query = tokenizer.from_list_format(
        [
                {"image": image_path},
                {"text": input_text}       
        ]
    )
    if if_add_template:
        raw_text, context_tokens = make_context(tokenizer, query)   # raw_text 自带 system prompt
    else:
        raw_text=query
    inputs = torch.tensor([tokenizer.encode(raw_text)]).to(model.device)
    inputs_embeds = build_inputs_embeds(model.base_model.transformer, inputs)
    attention_mask = torch.ones(inputs.shape, device=inputs.device)

    # 准备生成参数
    generated_ids = []
    eos_token_id = 151643   #"<|endoftext|>"
    model.eval()  # 切换到评估模式

    # 自回归生成循环
    for _ in range(max_new_tokens):
        # 前向传播
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # 获取下一个token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        if next_token.item() == eos_token_id:
            break
        # 记录生成token
        generated_ids.append(next_token.item())
        # 更新输入嵌入
        next_embed = model.base_model.transformer.wte(next_token)
        inputs_embeds = torch.cat([inputs_embeds, next_embed.unsqueeze(1)], dim=1)
        # 更新attention mask
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
    # 解码生成结果
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main(
    MODEL_NAME: str = "Qwen/Qwen-VL-Chat",
    data_path: str = '',
    save_dir: str = 'test_textual.csv',
    if_add_template: int=0,
    pt_weights: str= "../Saved_Tensors/Textual_400epo",
):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True, bf16=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = PeftModel.from_pretrained(
        model,
        pt_weights,
        torch_dtype=torch.bfloat16,
    )
    with open(data_path, "r") as f:
        datas=json.load(f)
    print(len(datas))
    outputs=[]
    outputs_json=[]
    for data in datas:
        response = infer(model, tokenizer, data,if_add_template)
        print(response)
        outputs.append(response)
        data['output']=response
        # print(data)
        outputs_json.append(data)
    df = pd.DataFrame(outputs)
    csv_dir=save_dir
    df.to_csv(csv_dir, index=False)
    # json_dir=save_dir+'/answer.json'
    # with open(json_dir, 'w') as f:
    #     json.dump(outputs_json, f, indent=4, separators=(", ", ": "), sort_keys=False)
fire.Fire(main)