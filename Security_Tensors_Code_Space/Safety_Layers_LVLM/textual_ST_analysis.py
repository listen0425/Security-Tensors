import sys
import fire
import torch
import json
import random
import torch.nn.functional as F
import copy
from tqdm import tqdm
from peft import PeftModel
import random
import matplotlib.pyplot as plt
import numpy as np
import random
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
# warnings.filterwarnings('ignore')

def get_r_lists_cossim(processor,model,model_adv,datapath1,seed,r=500):
    with open(datapath1, "r") as f:
        datas1=json.load(f)
    allcos=[]
    # plt.figure()
    for sss in range(r):
        random.seed(seed)
        seed=seed+1
        intermediate_outputs=[]
        data1=random.sample(datas1, 1)[0]
        input_text1=data1['input']
        prompt1="<|image|><|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text1+"\n\n### Response:\n"
        image_path1=data1['pic_path']
        image1 = Image.open(image_path1)
        inputs1 = processor(image1, prompt1, return_tensors="pt")
        all_vectors=[]
        generation_output1=infer(model, processor, image1, prompt1)
        generation_output2=model_adv.forward_orig(inputs1)
        hs1 = generation_output1['hidden_states']
        for i in range(len(hs1)):
            if i==0:
                continue
            all_vectors.append(hs1[i][0][-1])
        all_vectors2=[]
        hs2 = generation_output2['hidden_states']
        for i in range(len(hs2[0])):
            if i==0:
                continue
            all_vectors2.append(hs2[0][i][0][-1])
        cso=[]
        # print(len(all_vectors2),len(all_vectors))
        for k in range(len(all_vectors2)):
            a=all_vectors[k].cpu().detach().numpy()
            b=all_vectors2[k].cpu().detach().numpy()
            cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cso.append(cosine_similarity)  
        allcos.append(cso)
    print('end')
    return allcos



class AdversarialTuningLLaMAVision(nn.Module):
    def __init__(self, model_name):
        super(AdversarialTuningLLaMAVision, self).__init__()
        # type(self.processor): <class 'transformers.models.mllama.processing_mllama.MllamaProcessor'>
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MllamaForConditionalGeneration.from_pretrained(model_name,device_map='auto',torch_dtype=torch.float16,)
        self.model.eval()
        print(type(self.processor))

    def forward2(self, inputs, adver_tensor, max_new_tokens=1):
        co_inputs=copy.deepcopy(inputs)
        device=next(self.model.parameters()).device
        co_inputs.to(device)
        co_inputs["pixel_values"] = co_inputs["pixel_values"] + adver_tensor.to(device)
        # Forward pass through the model
        outputs = self.model.generate(**co_inputs, max_new_tokens=max_new_tokens,do_sample=False,output_hidden_states= True,
        return_dict_in_generate=True,
        output_scores=True,)
        return outputs
    def forward_orig(self, inputs, max_new_tokens=1):
        device=next(self.model.parameters()).device
        inputs.to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,do_sample=False,output_hidden_states= True,
        return_dict_in_generate=True,
        output_scores=True,)
        return outputs



def infer(model, processor, image, messages, max_new_tokens=32):
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
    for _ in range(1):
        with torch.no_grad():
            outputs = model(**inputs,output_hidden_states=True,return_dict=True,)
    # 解码生成结果
    return outputs


def main(
    normal_path: str='ana_data/normal.json',
    MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision",
    malicious_path: str='ana_data/malicious.json',
    save_dir: str='pkls/textual_tensor_activation.pkl',
    r: int=100,
    pt_weights: str = '../LLaMA-3.2-vision/Saved_Tensors/Textual_400epo',
    ):
    model_adv=AdversarialTuningLLaMAVision(MODEL_NAME)
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
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    device_map = 'auto'
    allcos_Normal_Norma_pairs=get_r_lists_cossim(processor,model,model_adv,normal_path,100,r)
    allcos_Mali_Mali_pairs=get_r_lists_cossim(processor,model,model_adv,malicious_path,100,r)
    allcos_Normal_Mali_pairs=get_r_lists_cossim(processor,model,model_adv,malicious_path,100,r)
    import pickle
    kkk=[allcos_Normal_Norma_pairs,allcos_Mali_Mali_pairs,allcos_Normal_Mali_pairs]
    # 使用Pickle写入

    sa=save_dir
    with open(sa, 'wb') as f:
        pickle.dump(kkk, f)
        
fire.Fire(main)