import torch, os
from transformers import MllamaForConditionalGeneration, AutoProcessor
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
import random
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM
)
from typing import List, Union
from PIL import Image
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

def get_logits_labels(prompt,processor,optimizer,model,inputs):
    device=next(model.parameters()).device
    label_str = extract_label_from_input(
        prompt, assistant_prefix="Response:\n"
    )
    label_ids = processor.tokenizer(
        text=label_str, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    optimizer.zero_grad()
    co_inputs=copy.deepcopy(inputs)
    outputs = model(**inputs)
    logits = outputs.logits
    # print(logits.size())
    # Shift logits and labels for loss computation
    shift_logits = logits[..., -label_ids.size(1)-1 : -1, :].contiguous().to(device)
    shift_labels = label_ids[..., :].contiguous().to(device)
    return shift_logits,shift_labels, outputs

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

def mask_labels(labels: torch.Tensor, separator_ids: list):
    # 在batch级别处理labels的掩码逻辑
    # Args:
    #     labels: 形状为(batch_size, seq_len)的输入张量
    #     separator_ids: 分隔符对应的token id列表
    # Returns:
    #     处理后的labels张量, 不符合条件的token位置设为-100

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

def main(
    MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision",
    data_path: str = '../../Dataset/LLaMA_vision.json',
    Epochs: int=1000,
    learning_rate: float=16e-4,
    tensor_save: str="../Saved_Tensors",
    keep_save: int=0,
    batch_size: int=1,
    tokens_num: int=100
):

    with open(data_path, "r") as f:
        datas=json.load(f)
    random.shuffle(datas)
    print(len(datas))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU device to use


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
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
    )
    for param in model.parameters():
        param.requires_grad = False
    
    model.language_model = get_peft_model(model.language_model, peft_config)
    # Llama 3.2 Error：word_embeddings is None
    if model.language_model.word_embeddings is None:
        model.language_model.word_embeddings = model.language_model.base_model.model.embed_tokens
    

    # print 可训练模块
    print("Trainable Modules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    LOSS=[]
    
    all=len(datas)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id)
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
            prompt="<|image|><|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text+"\n\n### Response:\n "+output_text+"<|end_of_text|>" 
            prompt_refuse="<|image|><|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text+"\n\n### Response:\n "+output_refuse+"<|end_of_text|>" 
            if image_type=='Harmful':   
                prompt_final=prompt_refuse
                inputs = processor(
                        images=image, text=prompt_refuse, return_tensors="pt", padding=True, truncation=True
                    ).to(model.language_model.device)
            else:
                prompt_final=prompt
                inputs = processor(
                        images=image, text=prompt, return_tensors="pt", padding=True, truncation=True
                    ).to(model.language_model.device)
            total_loss = 0
            for i in range(0, len(inputs["input_ids"]), batch_size):
                optimizer.zero_grad()
                batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
                # attention_mask 由 PeftModel 自动补齐, cross_attention_mask 需要手动补齐
                prefix_cross_attention_mask_shape = list(batch_inputs["cross_attention_mask"].shape)
                prefix_cross_attention_mask_shape[1] = peft_config.num_virtual_tokens
                prefix_cross_attention_mask = torch.zeros(
                    prefix_cross_attention_mask_shape,
                    dtype=batch_inputs["cross_attention_mask"].dtype,
                    device=batch_inputs["cross_attention_mask"].device,
                )
                batch_inputs["cross_attention_mask"] = torch.cat(
                    [
                        prefix_cross_attention_mask,
                        batch_inputs["cross_attention_mask"],
                    ],
                    dim=1,
                )
                shift_logits,shift_labels,outputs=get_logits_labels(prompt_final,processor,optimizer,model,batch_inputs)        
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
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3}, Avg Loss: {avg_loss:.4f}")
                decode_text = processor.decode(
                    outputs.logits.argmax(-1)[0], skip_special_tokens=False
                )
                # decode_text=extract_label_from_input(decode_text,assistant_prefix='Response:\n')
                print(f"Decoded text: {decode_text}")
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
                    df.to_csv(loss_dir, index=False)
fire.Fire(main)