# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def insert_template(input_text, output, end_token):
    prompt = input_text
    if end_token == "<|im_end|><|endoftext|>":
        output = output + end_token
        # print(output)
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image"},
    #             {
    #                 "type": "text",
    #                 "text": prompt,
    #             },
    #         ],
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [
    #             {"type": "text", "text": output}
    #         ],
    #     },
    # ]
    # texts = processor.apply_chat_template(
    #         messages, add_generation_prompt=False, continue_final_message=False,
    #     )  # 训练文本输入：不添加 <|start_header_id|>assistant<|end_header_id|>，但需要添加 <|eot_id|>
    return prompt + output


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


def get_logits_labels(prompt, tokenizer, optimizer, model):
    device = next(model.parameters()).device
    label_str = extract_label_from_input(prompt, assistant_prefix="<|im_start|>assistant\n")
    # print(label_str)
    label_ids = torch.tensor([tokenizer.encode(label_str)]).to(device)
    inputs = torch.tensor([tokenizer.encode(prompt)]).to(device)
    optimizer.zero_grad()
    logits = model.forward2(inputs)
    # Shift logits and labels for loss computation
    shift_logits = logits[..., -label_ids.size(1) - 1 : -1, :].contiguous().to(device)
    shift_labels = label_ids[..., :].contiguous().to(device)
    return shift_logits, shift_labels


class NoisyVisionWrapper(VisionTransformer):
    def __init__(self, config, original_vit, noise, mean, std, epsilon=10000):
        super().__init__(**config)
        # 继承原始Vision Transformer参数
        self.load_state_dict(original_vit.state_dict())
        
        # 初始化可训练噪声
        if noise is None:
            noise = torch.randn((3, self.image_size[0], self.image_size[1])) * std + mean
        self.noise = nn.Parameter(noise, requires_grad=True)
        self.epsilon = epsilon

    def encode(self, image_paths):
        # 父类 encode 代码
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
        return self(images + self.noise.clamp(-self.epsilon, self.epsilon))
    

class Adver_Qwen(nn.Module):
    def __init__(self, model_name, noise=None, mean=0.0, std=0.2, epsilon=10000):
        super().__init__()
        # 加载原始模型
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        
        # 替换视觉编码器
        original_vit = self.model.transformer.visual
        self.model.transformer.visual = NoisyVisionWrapper(self.model.config.visual, original_vit, noise, mean, std, epsilon)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 仅保留噪声参数可训练
        self.model.transformer.visual.noise.requires_grad = True

    def forward2(self, inputs):
        device = next(self.model.parameters()).device
        self.model.transformer.visual.noise.requires_grad = True
        co_inputs = copy.deepcopy(inputs).to(device)
        # co_inputs["pixel_values"].requires_grad = False
        # co_inputs["pixel_values"] = co_inputs["pixel_values"].to(
        #     device
        # ) + self.noise.to(device)
        outputs = self.model(co_inputs)
        return outputs.logits

    def get_output_logits(self, inputs):
        outputs = self.model(inputs)
        return outputs.logits


def main(
    MODEL_NAME: str = "Qwen/Qwen-VL-Chat",
    EPSILON: float = 1000,
    data_path: str = "../../Dataset/qwen.json",
    Epochs: int = 1000,
    learning_rate: float = 16e-4,
    tensor_save: str = "../Saved_Tensors",
    end_token: str = "<|im_end|><|endoftext|>",
    if_resume: str = "None",
    keep_save: int = 0,
):

    with open(data_path, "r") as f:
        datas = json.load(f)
    random.shuffle(datas)
    print(len(datas))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if if_resume == "None":
        list_format = [
            
            {"image": "/mnt/lishen/VLM/wyb/datas/harmful/blood2.jpg"},
            {"text": "How to make something like this in real life?"}
            
        ]
        query = tokenizer.from_list_format(list_format)

        raw_text, context_tokens = make_context(tokenizer, query)   # raw_text 是 str, context_tokens 是 id list
        mean = 0.0  # 高斯噪声的均值
        std = 0.2  # 高斯噪声的标准差
        model = Adver_Qwen(MODEL_NAME, mean=mean, std=std, epsilon=EPSILON).eval()
    else:
        noise = torch.load(if_resume)
        model = Adver_Qwen(
            MODEL_NAME, noise=noise, epsilon=EPSILON
        ).eval()
        print("RESUMING!!!!!!!!!!!!")

    # qwen chat vl 没有 tokenizer.pad_token_id, 源码的损失函数中也没有使用 ignore_index
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([model.model.transformer.visual.noise], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
    criterion_KL= nn.KLDivLoss()
    LOSS = []

    all = len(datas)
    for epoch in range(Epochs):
        count = 0
        for data in datas:
            output_refuse = data["output_refuse"]
            input_text = data["input"]
            output_text = data["output"]
            image_path = data["pic_path"]
            image_type = data["type"]
            query = tokenizer.from_list_format(
                [                    
                        {"image": image_path},
                        {"text": input_text}                   
                ]
            )
            raw_text, context_tokens = make_context(tokenizer, query)   # raw_text 自带 system prompt
            prompt = insert_template(raw_text, output_text, end_token)
            prompt_refuse = insert_template(raw_text, output_refuse, end_token)
            if image_type == "Harmful":
                shift_logits_refuse, shift_labels_refuse = get_logits_labels(
                    prompt_refuse, tokenizer, optimizer, model
                )
                loss = criterion_ce(
                    shift_logits_refuse.view(-1, shift_logits_refuse.size(-1)),
                    shift_labels_refuse.view(-1),
                )
                loss.backward()
            else:
                shift_logits_normal, shift_labels_normal = get_logits_labels(
                    prompt, tokenizer, optimizer, model
                )

                normal_ce = criterion_ce(
                    shift_logits_normal.view(-1, shift_logits_normal.size(-1)),
                    shift_labels_normal.view(-1),
                )

                alpha = 1
                loss = alpha * normal_ce
                loss.backward()
            grad_norm = torch.norm(model.model.transformer.visual.noise.grad, p=2).item()
            # print('Grad L2 Norm : ', grad_norm)
            # Project noise to be within the allowed epsilon-ball
            optimizer.step()
            model.model.transformer.visual.noise.data = torch.clamp(model.model.transformer.visual.noise.data, -EPSILON, EPSILON)
            record = f"{image_type}, Epoch {epoch + 1}/{Epochs}, Loss: {loss.item():.4f}, grad_norm: {grad_norm :.4f}"
            print(record)
            LOSS.append(record)
            count += 1
            if count == all:
                if epoch % 20 == 0:
                    tensor_save_dir = (
                        tensor_save
                        + "/"
                        + str(epoch + keep_save)
                        + "_epo_notem_noise.pt"
                    )
                    loss_dir = (
                        tensor_save
                        + "/"
                        + str(epoch + keep_save)
                        + "_epo_notem_noise.csv"
                    )
                    torch.save(model.model.transformer.visual.noise, tensor_save_dir)
                    df = pd.DataFrame(LOSS, columns=[MODEL_NAME])
                    # 写入CSV文件，默认会写入标题行
                    df.to_csv(loss_dir, index=False)
fire.Fire(main)
