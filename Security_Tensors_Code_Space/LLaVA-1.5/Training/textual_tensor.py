import torch, os
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
from typing import List, Union
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
import copy
import json
import random
import pandas as pd
import fire
import torch.optim as optim

def kl_shift_orig(prompt,processor,image,optimizer,model):
    label_str = extract_label_from_input(
        prompt, assistant_prefix="Response:\n"
    )
    label_ids = processor.tokenizer(
        text=label_str, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    inputs_orig = processor(
        image, prompt, return_tensors="pt", padding=True, truncation=True
    )
    logits_orig_model=model.get_output_logits(inputs_orig)
    shift_logits_orig_model = logits_orig_model[..., -label_ids.size(1)-1 : -1, :].contiguous()
    probs_orig=torch.softmax(shift_logits_orig_model,dim=-1)
    return probs_orig


def get_logits_labels(prompt,processor,optimizer,model,inputs,attention_mask):
    device=next(model.parameters()).device
    label_str = extract_label_from_input(
        prompt, assistant_prefix="ASSISTANT:"
    )
    label_ids = processor.tokenizer(
        text=label_str, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    optimizer.zero_grad()
    co_inputs=copy.deepcopy(inputs)
    outputs = model.language_model(attention_mask=attention_mask, inputs_embeds=co_inputs)
    logits = outputs.logits
    # print(logits.size())
    # Shift logits and labels for loss computation
    shift_logits = logits[..., -label_ids.size(1)-1 : -1, :].contiguous().to(device)
    shift_labels = label_ids[..., :].contiguous().to(device)
    return shift_logits,shift_labels, outputs


def mask_labels(labels: torch.Tensor, separator_ids: list):
    """
    在batch级别处理labels的掩码逻辑
    Args:
        labels: 形状为(batch_size, seq_len)的输入张量
        separator_ids: 分隔符对应的token id列表
    Returns:
        处理后的labels张量, 不符合条件的token位置设为-100
    """
    batch_size, seq_len = labels.shape
    k = len(separator_ids)
    if k == 0 or seq_len < k:
        labels.fill_(-100)
        return labels
    # 将分隔符转换为设备兼容的tensor
    separator = torch.tensor(separator_ids, device=labels.device)
    # 生成滑动窗口视图 (batch_size, seq_len-k+1, k)
    windows = labels.unfold(1, k, 1)
    # 匹配分隔符模式 (batch_size, seq_len-k+1)
    matches = (windows == separator.view(1, 1, -1)).all(dim=2)
    # 创建位置索引并计算匹配分数
    positions = torch.arange(seq_len-k+1, device=labels.device).expand(batch_size, -1)
    scores = matches.float() * positions.float()
    # 获取最后一个匹配位置
    max_scores, max_indices = torch.max(scores, dim=1)
    start_positions = torch.where(max_scores > 0, max_indices, -1)
    # 生成掩码
    pos = torch.arange(seq_len, device=labels.device).view(1, -1)
    cutoff = (start_positions + k).view(-1, 1)
    mask = (pos < cutoff) | (start_positions == -1).view(-1, 1)
    # 应用掩码
    labels.masked_fill_(mask, -100)
    return labels


def insert_template(processor,input_text,output,end_token,add_res=1):
    if add_res==1:
        prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text+"\n\n### Response:\n "
    else:
        prompt=input_text
    if end_token=='</s>':
        output=output+end_token
        # print(output)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": output}
            ],
        },
    ]
    texts = processor.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=False,
        )  # 训练文本输入：不添加 <|start_header_id|>assistant<|end_header_id|>，但需要添加 <|eot_id|>
    texts =  "USER: <image>\n"+prompt+"ASSISTANT:"+output
    return texts

def extract_label_from_input(processed_input, assistant_prefix="ASSISTANT:"):
    # processed_input 是 processor.apply_chat_template 的输出
    # 找到 `ASSISTANT:` 的位置
    assistant_start = processed_input.find(assistant_prefix)
    if assistant_start != -1:
        # 提取 `ASSISTANT:` 后的内容并去掉多余空格
        label = processed_input[assistant_start + len(assistant_prefix) :].strip()
        return label
    else:
        raise ValueError("ASSISTANT prefix not found in input!")

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
    
def main(
    MODEL_NAME: str = "llava-hf/llava-1.5-7b-hf",
    EPSILON : float = 1000,
    data_path: str = '../../Dataset/llava.json',
    Epochs: int=1000,
    tensor_save: str="../Saved_Tensors",
    end_token: str="</s>",
    if_resume: str="None",
    keep_save: int=0,
    add_res=1,
    learning_rate: float=1e-3,
    batch_size: int=1,
    tokens_num: int=10,
):

    with open(data_path, "r") as f:
        datas=json.load(f)
    random.shuffle(datas)
    print(len(datas))
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM, # This type indicates the model will generate text.
        prompt_tuning_init=PromptTuningInit.RANDOM,  # PromptTuningInit.RANDOM or PromptTuningInit.TEXT
        prompt_tuning_init_text=None, # The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        num_virtual_tokens=tokens_num, # Number of virtual tokens to be added and trained.
        tokenizer_name_or_path=MODEL_NAME, # The tokenizer used for the model.
    )

# 模型
# RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
    model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
        )
    for param in model.parameters():
        param.requires_grad = False
    model.language_model = get_peft_model(model.language_model, peft_config)

    # print 可训练模块
    print("Trainable Modules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    LOSS=[]
    all=len(datas)
    model.train()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
    criterion_KL= nn.KLDivLoss()
    for epoch in range(Epochs):
        count=0
        for data in datas:
            output_refuse=data['output_refuse']
            input_text=data['input']
            output_text=data['output']
            image_path=data['pic_path']
            image_type=data['type']
            image = Image.open(image_path)
            prompt=insert_template(processor,input_text,output_text,end_token,add_res=add_res)
            prompt_refuse=insert_template(processor,input_text,output_refuse,end_token,add_res=add_res)

            for name, param in model.named_parameters():
                if name=='language_model.prompt_encoder.default.embedding.weight':
                    param.requires_grad=True
                    break
            # print(prompt_refuse)
            if image_type=='Harmful':   
                prompt=prompt_refuse
                inputs = processor(
                        images=image, text=prompt_refuse, return_tensors="pt", padding=True, truncation=True
                    ).to(model.language_model.device)
                image_features = model.get_image_features(
                    pixel_values=inputs["pixel_values"],
                    vision_feature_layer=model.config.vision_feature_layer,  # 使用模型配置中的默认值
                    vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    # image_sizes=,
                )
                # 构建 inputs_embeds
                inputs_embeds = build_inputs_embeds(model, inputs["input_ids"], image_features)
            else:
                inputs = processor(
                        images=image, text=prompt, return_tensors="pt", padding=True, truncation=True
                    ).to(model.language_model.device)
                image_features = model.get_image_features(
                    pixel_values=inputs["pixel_values"],
                    vision_feature_layer=model.config.vision_feature_layer,  # 使用模型配置中的默认值
                    vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    # image_sizes=,
                )
                # 构建 inputs_embeds
                inputs_embeds = build_inputs_embeds(model, inputs["input_ids"], image_features)
            total_loss = 0
            for i in range(0, len(inputs["input_ids"]), batch_size):
                optimizer.zero_grad()
                input_ids = inputs["input_ids"][i:i + batch_size]
                attention_mask = inputs["attention_mask"][i:i + batch_size]
                inputs_embeds = inputs_embeds[i:i + batch_size]
                shift_logits,shift_labels,outputs=get_logits_labels(prompt,processor,optimizer,model,inputs_embeds,attention_mask)        
                loss = criterion_ce(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                # kl_orig=kl_shift_orig(prompt,processor,image,optimizer,model)
                # log_kl=torch.log_softmax(shift_logits,dim=-1)
                # loss_kl = criterion_KL(
                #     log_kl.view(-1, log_kl.size(-1)), kl_orig.view(-1, kl_orig.size(-1))
                # )
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(inputs["input_ids"])
            # Project noise to be within the allowed epsilon-ball
            record=f"{image_type}, Epoch {epoch + 1}/{Epochs}, Loss: {avg_loss:.4f}"
            print(record)
            LOSS.append(record)
            count+=1
            if count==all:
                if epoch %20==0:
                    output_dir = tensor_save+'/'+str(epoch)
                    model.language_model.save_pretrained(output_dir, safe_serialization=False)
                    loss_dir=output_dir+'/'+str(epoch+keep_save)+'_epo_notem_noise.csv'
                    df = pd.DataFrame(LOSS, columns=[MODEL_NAME])
                    # 写入CSV文件，默认会写入标题行
                    df.to_csv(loss_dir, index=False)
fire.Fire(main)
