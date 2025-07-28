import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    MllamaImageProcessor,
    AutoProcessor,
    AutoModelForCausalLM
)
import json
import random
import pandas as pd
import fire

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

class AdversarialTuningLLaMAVision(nn.Module):
    def __init__(self, model_name):
        super(AdversarialTuningLLaMAVision, self).__init__()
        # type(self.processor): <class 'transformers.models.mllama.processing_mllama.MllamaProcessor'>
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MllamaForConditionalGeneration.from_pretrained(model_name,device_map='auto',torch_dtype=torch.float16,)
        self.model.eval()

        print(type(self.processor))

    def forward2(self, inputs, adver_tensor, max_new_tokens=50):
        device=next(self.model.parameters()).device
        inputs.to(device)
        inputs["pixel_values"] = inputs["pixel_values"] + adver_tensor.to(device)
        # Forward pass through the model
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,do_sample=False)
        return outputs
    def forward_orig(self, inputs, max_new_tokens=50):
        device=next(self.model.parameters()).device
        inputs.to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,do_sample=False)
        return outputs

def main(
    MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision",
    data_path: str = '',
    save_dir: str = 'test_visual.csv',
    adver_path: str = '../Saved_Tensors/Visual_400epo.pt',
    if_add_tensor: int = 1,
):

    with open(data_path, "r") as f:
        datas=json.load(f)
    print(len(datas))
    adver_tensor=torch.load(adver_path)
    model = AdversarialTuningLLaMAVision(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    outputs=[]
    for data in datas:
        input_text=data['input']
        prompt="<|image|><|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text+"\n\n### Response:\n"
        image_path=data['pic_path']
        # image_type=data['type']
        # if image_type!="Harmless":
        #     continue
        image = Image.open(image_path)
        inputs = processor(image, prompt, return_tensors="pt")
        print("*****"*10)
        if if_add_tensor==1:
            output = model.forward2(inputs,adver_tensor)
        else:
            output=model.forward_orig(inputs)
        print(processor.decode(output[0]))
        outputs.append(processor.decode(output[0]))
        
    df = pd.DataFrame(outputs)
    df.to_csv(save_dir, index=False)
    
fire.Fire(main)