import torch, os
from transformers import LlavaForConditionalGeneration, AutoProcessor,  MllamaForConditionalGeneration
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

def insert_template(input_text):
    texts = "<|image|><|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text+"\n\n### Response:\n"
    return texts

def infer(model, processor, image, messages, max_new_tokens=16):
    # 参数说明:
    # add_generation_prompt 是否在对话末尾添加生成提示符（例如 <|start_header_id|>assistant<|end_header_id|>）, 指示模型开始生成回复。
    # texts = processor.apply_chat_template(
    #     messages, add_generation_prompt=True  # 添加assistant起始标记
    # )
    # print("Full texts:", texts)
    inputs = processor(
        images=image, text=messages, return_tensors="pt", padding=True, truncation=True
    ).to(model.language_model.device)
    # print(inputs.keys())    # ['input_ids', 'attention_mask', 'pixel_values']
    # print(inputs['input_ids'].shape)
    # print(processor.decode(inputs["input_ids"][0]))

    # 处理cross_attention_mask（添加虚拟token前缀）
    prefix_cross_attention_mask_shape = list(inputs["cross_attention_mask"].shape)
    prefix_cross_attention_mask_shape[1] = model.language_model.peft_config['default'].num_virtual_tokens
    prefix_cross_attention_mask = torch.zeros(
        prefix_cross_attention_mask_shape,
        dtype=inputs["cross_attention_mask"].dtype,
        device=inputs["cross_attention_mask"].device,
    )
    inputs["cross_attention_mask"] = torch.cat(
        [
            prefix_cross_attention_mask,
            inputs["cross_attention_mask"],
        ],
        dim=1,
    )

    # 准备生成参数
    generated_ids = []
    eos_token_id = "<|end_of_text|>" 
    model.eval()  # 切换到评估模式

    # 自回归生成循环
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取下一个token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        if next_token.item() == eos_token_id:
            break

        # 记录生成token
        generated_ids.append(next_token.item())

        # 更新输入嵌入
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token.unsqueeze(0)], dim=1
        )

        # 更新attention mask
        inputs["attention_mask"] = torch.cat(
            [
                inputs["attention_mask"],
                inputs["attention_mask"].new_ones((inputs["attention_mask"].shape[0], 1)),
            ],
            dim=-1,
        )

        # 更新cross_attention_mask
        # print(inputs["cross_attention_mask"].shape)
        inputs["cross_attention_mask"] = torch.cat(
            [
                inputs["cross_attention_mask"],
                inputs["cross_attention_mask"][:, -1:, ...],
            ],
            dim=1,
        )

    # 解码生成结果
    return processor.decode(generated_ids, skip_special_tokens=True)

def main(
    MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision",
    data_path: str = '',
    pt_weights: str = "../Saved_Tensors/Textual_400epo",
    save_dir: str = 'test_textual.csv',
):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        pt_weights,
        torch_dtype=torch.float16,
    )
    # Llama 3.2 Error：word_embeddings is None
    if model.language_model.word_embeddings is None:
        model.language_model.word_embeddings = (
            model.language_model.base_model.model.embed_tokens
        )
    with open(data_path, "r") as f:
        datas=json.load(f)
    print(len(datas))
    outputs=[]
    outputs_json=[]
    for data in datas:
        input_text=data['input']
        prompt=insert_template(input_text)
        image_path=data['pic_path']
        image = Image.open(image_path)
        result = infer(model, processor, image, prompt)
        print(result)
        outputs.append(result)
        
    df = pd.DataFrame(outputs)
    df.to_csv(save_dir, index=False)
    
fire.Fire(main)