import torch, os
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM
)
import json
import random
import pandas as pd
import fire

def insert_template(input_text,if_add_prompt_template):
    if if_add_prompt_template:
        prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text+"\n\n### Response:\n "
    else:
        prompt=input_text
    texts =  "USER: <image>\n"+prompt+"ASSISTANT:"
    return texts

def build_inputs_embeds(
    model,
    input_ids: torch.LongTensor,
    image_features: torch.Tensor,
):
    # Step 1: 获取原始文本的 Embeddings
    inputs_embeds = model.get_input_embeddings()(input_ids)  # (batch_size, seq_len, hidden_size)

    # Step 2: 定位 <image> 标记的位置
    special_image_mask = (input_ids == model.config.image_token_index).unsqueeze(-1)
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

    # Step 3: 验证图像特征与标记数量匹配
    if inputs_embeds[special_image_mask].numel() != image_features.numel():
        n_image_tokens = (input_ids == model.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        raise ValueError(
            f"Image features ({n_image_features}) 与 image tokens ({n_image_tokens}) 数量不匹配"
        )

    # Step 4: 将图像特征嵌入到 inputs_embeds 中
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # # Step 5: 获取 Prompt Embeddings
    # prompts = model.language_model.get_prompt(batch_size=text_embeds.shape[0]).to(text_embeds.dtype)

    # # Step 6: 拼接 Prompt Embeddings
    # inputs_embeds = torch.cat((prompts, text_embeds), dim=1)  # (batch_size, prompt_len + seq_len, hidden_size)
    return inputs_embeds

def infer(model, processor, image, texts, max_new_tokens=32):
    # 参数说明: 
    # add_generation_prompt 是否在对话末尾添加生成提示符（例如 <|start_header_id|>assistant<|end_header_id|>）, 指示模型开始生成回复。
    # 推理时 llava-1.5-7b-hf/tokenizer_config.json "add_eos_token" 改为 false, 不需要结束符 </s>
    # texts = processor.apply_chat_template(
    #     messages, add_generation_prompt=True  # 添加assistant起始标记
    # )

    inputs = processor(
        images=image, text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.language_model.device)
    
    # 提取图像特征（使用整个模型的视觉编码器）
    image_features = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )
    
    # 构建初始输入嵌入（融合图像+文本）
    inputs_embeds = build_inputs_embeds(model, inputs["input_ids"], image_features)
    attention_mask = inputs["attention_mask"]
    
    # 准备生成参数
    generated_ids = []
    eos_token_id = processor.tokenizer.eos_token_id
    model.eval()  # 切换到评估模式
    
    # 自回归生成循环
    for _ in range(max_new_tokens):
        # 前向传播（仅使用语言模型）
        outputs = model.language_model(
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
        next_embed = model.language_model.get_input_embeddings()(next_token)
        inputs_embeds = torch.cat([inputs_embeds, next_embed.unsqueeze(1)], dim=1)
        
        # 更新attention mask
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((1, 1), device=attention_mask.device)
        ], dim=1)
    
    # 解码生成结果
    return processor.decode(generated_ids, skip_special_tokens=True)


def main(
    MODEL_NAME: str = "llava-hf/llava-1.5-7b-hf",
    data_path: str = '',
    pt_weights: str = "../Saved_Tensors/Textual_400epo",
    save_dir: str = 'test_textual.csv',
    if_add_prompt_template: int=1,
    if_add_tensor: int = 0,
):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        pt_weights,
        torch_dtype=torch.float16,
    )
    with open(data_path, "r") as f:
        datas=json.load(f)
    print(len(datas))
    outputs=[]
    outputs_json=[]
    for data in datas:
        input_text=data['input']
        prompt=insert_template(input_text,if_add_prompt_template)
        image_path=data['pic_path']
        image = Image.open(image_path)
        result = infer(model, processor, image, prompt)
        print(result)
        outputs.append(result)
        
    df = pd.DataFrame(outputs)
    df.to_csv(save_dir, index=False)
    
fire.Fire(main)