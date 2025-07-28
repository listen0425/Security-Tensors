import sys
import fire
import torch
import json
import random
import torch.nn.functional as F
from tqdm import tqdm
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

def get_r_lists_cossim(processor,model,adver_tensor,datapath1,datapath2,seed,r=500):
    with open(datapath1, "r") as f:
        datas1=json.load(f)
    with open(datapath2, "r") as f:
        datas2=json.load(f)

    allcos=[]
    # plt.figure()
    for sss in range(r):
        random.seed(seed)
        seed=seed+1
        intermediate_outputs=[]
        if datapath1 == datapath2:
            random_items = random.sample(datas1, 2)
            data1=random_items[0]
            data2=random_items[1]
        else:
            data1=random.sample(datas1, 1)[0]
            data2=random.sample(datas2, 1)[0]
        input_text1=data1['input']
        prompt1="<|image|><|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text1+"\n\n### Response:\n"
        image_path1=data1['pic_path']
        image1 = Image.open(image_path1)
        inputs1 = processor(image1, prompt1, return_tensors="pt")
        input_text2=data2['input']
        prompt2="<|image|><|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text2+"\n\n### Response:\n"
        image_path2=data2['pic_path']
        image2 = Image.open(image_path2)
        inputs2 = processor(image2, prompt2, return_tensors="pt")
        all_vectors=[]
        if adver_tensor!='None':
            generation_output1=model.forward2(inputs1,adver_tensor)
        else:
            generation_output1=model.forward_orig(inputs1)
        hs1 = generation_output1['hidden_states']
        for i in range(len(hs1[0])):
            if i==0:
                continue
            all_vectors.append(hs1[0][i][0][-1])
            
        all_vectors2=[]
        if adver_tensor!='None':
            generation_output2=model.forward2(inputs2,adver_tensor)
        else:
            generation_output2=model.forward_orig(inputs2)
        hs2 = generation_output2['hidden_states']
        for i in range(len(hs2[0])):
            if i==0:
                continue
            all_vectors2.append(hs2[0][i][0][-1])
        cso=[]
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
        device=next(self.model.parameters()).device
        inputs.to(device)
        inputs["pixel_values"] = inputs["pixel_values"] + adver_tensor.to(device)
        # Forward pass through the model
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,do_sample=False,output_hidden_states= True,
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
    
def main(
    normal_path: str='ana_data/normal.json',
    MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision",
    malicious_path: str='ana_data/malicious.json',
    save_dir: str='pkls/llama32_multi-modal_infer.pkl',
    r: int=100,
    ):
    model = AdversarialTuningLLaMAVision(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    allcos_Normal_Norma_pairs=get_r_lists_cossim(processor,model,'None',normal_path,normal_path,100,r)
    allcos_Mali_Mali_pairs=get_r_lists_cossim(processor,model,'None',normal_path,malicious_path,100,r)
    allcos_Normal_Mali_pairs=get_r_lists_cossim(processor,model,'None',normal_path,malicious_path,100,r)
    import pickle
    kkk=[allcos_Normal_Norma_pairs,allcos_Mali_Mali_pairs,allcos_Normal_Mali_pairs]
    # 使用Pickle写入
    with open(save_dir, 'wb') as f:
        pickle.dump(kkk, f)
        
fire.Fire(main)