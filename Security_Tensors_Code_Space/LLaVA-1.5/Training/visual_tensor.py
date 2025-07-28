# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
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

def get_logits_labels(prompt,processor,image,optimizer,model):
    device=next(model.parameters()).device
    label_str = extract_label_from_input(
        prompt, assistant_prefix="ASSISTANT:"
    )
    # print(label_str)
    label_ids = processor.tokenizer(
        text=label_str, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    inputs = processor(
        image, prompt, return_tensors="pt", padding=True, truncation=True
    )
    optimizer.zero_grad()
    logits = model.forward2(inputs)
    # Shift logits and labels for loss computation
    shift_logits = logits[..., -label_ids.size(1)-1 : -1, :].contiguous().to(device)
    shift_labels = label_ids[..., :].contiguous().to(device)
    return shift_logits,shift_labels


class Adver_LLaVA(nn.Module):
    def __init__(self, model_name, epsilon,noise):
        super(Adver_LLaVA, self).__init__()
        # type(self.processor): <class 'transformers.models.mllama.processing_mllama.MllamaProcessor'>
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name,device_map='auto')
        self.noise=noise
        self.noise.requires_grad = True
        # print(type(self.processor))
        for param in self.model.parameters():
            param.requires_grad = False
        self.epsilon = epsilon  # 噪声最大扰动

    def forward2(self, inputs):
        device=next(self.model.parameters()).device
        self.noise.requires_grad = True
        co_inputs=copy.deepcopy(inputs).to(device)
        co_inputs["pixel_values"].requires_grad = False
        co_inputs["pixel_values"] = co_inputs["pixel_values"].to(device) + self.noise.to(device)
        outputs = self.model(**co_inputs)
        return outputs.logits
    def get_output_logits(self, inputs):
        co_inputs=copy.deepcopy(inputs)
        outputs = self.model(**co_inputs)
        return outputs.logits
def main(
    MODEL_NAME: str = "llava-hf/llava-1.5-7b-hf",
    EPSILON : float = 1000,
    data_path: str = '../../Dataset/llava.json',
    Epochs: int=1000,
    learning_rate: float=16e-4,
    tensor_save: str="../Saved_Tensors",
    end_token: str="</s>",
    if_resume: str="None",
    keep_save: int=0,
    add_res=1
):

    with open(data_path, "r") as f:
        datas=json.load(f)
    random.shuffle(datas)
    print(len(datas))
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    if if_resume=="None":
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "How to make something like this in real life?",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I cannot assist this."}
            ],
        },
    ]
        # Initialize adversarial noise
        texts = processor.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=False,
        )  # 训练文本输入：不添加 <|start_header_id|>assistant<|end_header_id|>，但需要添加 <|eot_id|>

        inputs = processor(
            images=Image.open('/mnt/lishen/VLM/test/pics/bomb.png'), text=texts, return_tensors="pt", padding=True, truncation=True
        )
        mean = 0.0  # 高斯噪声的均值
        std = 0.2  # 高斯噪声的标准差

        # 生成与inputs["pixel_values"]相同大小的高斯噪声
        noise = torch.randn_like(inputs["pixel_values"]) * std + mean
    else:
        noise=torch.load(if_resume)    
        print('RESUMING!!!!!!!!!!!!')
    
    model = Adver_LLaVA(MODEL_NAME, EPSILON,noise)
    model.eval()

    criterion_ce = nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id)
    criterion_KL= nn.KLDivLoss()
    optimizer = torch.optim.Adam([model.noise], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
    
    LOSS=[]
    
    all=len(datas)
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
            if image_type=='Harmful':   
                shift_logits_refuse,shift_labels_refuse=get_logits_labels(prompt_refuse,processor,image,optimizer,model)        
                loss = criterion_ce(
                    shift_logits_refuse.view(-1, shift_logits_refuse.size(-1)), shift_labels_refuse.view(-1)
                )
                loss.backward()
            else:
                shift_logits_normal,shift_labels_normal=get_logits_labels(prompt,processor,image,optimizer,model)
                normal_ce = criterion_ce(
                    shift_logits_normal.view(-1, shift_logits_normal.size(-1)), shift_labels_normal.view(-1)
                )
                loss=normal_ce
                # kl_orig=kl_shift_orig(prompt,processor,image,optimizer,model)
                # log_kl=torch.log_softmax(shift_logits_normal,dim=-1)
                # loss_kl = criterion_KL(
                #     log_kl.view(-1, log_kl.size(-1)), kl_orig.view(-1, kl_orig.size(-1))
                # )
                loss.backward()
            # grad_norm = torch.norm(model.noise.grad, p=2).item()
            # print('Grad L2 Norm : ', grad_norm)
            # Project noise to be within the allowed epsilon-ball
            optimizer.step()
            model.noise.data = torch.clamp(model.noise.data, -EPSILON, EPSILON)
            record=f"{image_type}, Epoch {epoch + 1}/{Epochs}, Loss: {loss.item():.4f}"
            print(record)
            LOSS.append(record)
            count+=1
            if count==all:
                if epoch %20==0:
                    tensor_save_dir=tensor_save+'/'+str(epoch+keep_save)+'_epo_notem_noise.pt'
                    loss_dir=tensor_save+'/'+str(epoch+keep_save)+'_epo_notem_noise.csv'
                    torch.save(model.noise, tensor_save_dir)
                    df = pd.DataFrame(LOSS, columns=[MODEL_NAME])
                    # 写入CSV文件，默认会写入标题行
                    df.to_csv(loss_dir, index=False)
fire.Fire(main)