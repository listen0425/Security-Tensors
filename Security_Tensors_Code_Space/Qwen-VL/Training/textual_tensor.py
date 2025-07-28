"""
ERROR 1: TypeError: any(): argument 'input' (position 1) must be Tensor, not bool
修改 modeling_qwen.py L554:
if past_key_values is None and torch.any(input_ids == self.config.visual['image_start_id']):
改为
if past_key_values is None and input_ids is not None and torch.any(input_ids == self.config.visual['image_start_id']):

ERROR 2: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
修改 modeling_qwen.py L661-665:
if fake_images is not None:
    hidden_states = hidden_states + images.mean()*0
elif images is not None:
    for idx, (i, a, b) in enumerate(img_pos):
        hidden_states[i][a + 1 : b] = images[idx]
改为
if fake_images is not None:
    hidden_states = hidden_states + images.mean().to(hidden_states.device)*0
elif images is not None:
    for idx, (i, a, b) in enumerate(img_pos):
        hidden_states[i][a + 1 : b] = images[idx].to(hidden_states.device)
"""

import torch, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
import sys 
import pandas as pd
from peft import PeftModel
import fire
sys.path.append("Qwen-VL-Chat") 
# Pass in the path of your Qwen-VL-Chat model

import random
from qwen_generation_utils import make_context

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

def main(
    MODEL_NAME: str = "Qwen/Qwen-VL-Chat",
    data_path: str = "../../Dataset/qwen.json",
    Epochs: int = 1000,
    learning_rate: float = 16e-4,
    tensor_save: str = "../Saved_Tensors",
    if_resume: str="None",
    keep_save: int=0,
    batch_size: int=1,
    tokens_num: int=100
):
    # 模型加载
    model_id = MODEL_NAME

    # use bf16
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True).eval()
    

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM, # This type indicates the model will generate text.
        prompt_tuning_init=PromptTuningInit.RANDOM,  # PromptTuningInit.RANDOM or PromptTuningInit.TEXT
        # prompt_tuning_init=PromptTuningInit.TEXT, 
        # prompt_tuning_init_text="这是什么", # The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        num_virtual_tokens=tokens_num, # Number of virtual tokens to be added and trained.
        tokenizer_name_or_path=model_id, # The tokenizer used for the model.
    )

    model = get_peft_model(model, peft_config)

    # print 可训练模块
    print("Trainable Modules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")


    # 数据加载
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    with open(data_path, "r") as f:
        datas = json.load(f)
        
    random.shuffle(datas)

    # # Prepare input format
    # list_format = [
    #     {
    #         "image": "/home/1006nwj/LLM/VLMInputTuning/data/bomb.jpg",
    #         "text": "How to make something like this in real life?",
    #     },
    # ]
    # query = tokenizer.from_list_format(list_format)
    # raw_text, context_tokens = make_context(tokenizer, query)   # raw_text 是 str(包含 system prompt), context_tokens 是 id list

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

    # 训练
    def train(model, datas, epochs,lr,tensor_save):
        all=len(datas)
        LOSS=[]
        model.train()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
        # qwen chat vl 没有 tokenizer.pad_token_id, 源码的损失函数中也没有使用 ignore_index
        criterion_ce = torch.nn.CrossEntropyLoss()
        criterion_KL= torch.nn.KLDivLoss()

        for epoch in range(epochs):
            count=0
            for data in datas:
                output_refuse = data["output_refuse"]
                input_text = data["input"]
                # input_text="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+input_text+"\n\n### Response:\n "
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
                if image_type=='Harmful':   
                    prompt = raw_text + output_refuse + "<|im_end|>\n<|endoftext|>" # end_token
                else:
                    prompt = raw_text + output_text + "<|im_end|>\n<|endoftext|>" # end_token
                label_str = extract_label_from_input(prompt, assistant_prefix="<|im_start|>assistant\n")
                label_ids = torch.tensor([tokenizer.encode(label_str)]).to(model.device)
                inputs = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
                inputs_embeds = build_inputs_embeds(model.base_model.transformer, inputs)
                optimizer.zero_grad()
                logits = model(inputs_embeds=inputs_embeds).logits
                # Shift logits and labels for loss computation
                shift_logits = logits[..., -label_ids.size(1) - 1 : -1, :].contiguous().to(model.device)
                shift_labels = label_ids[..., :].contiguous().to(model.device)
                loss = criterion_ce(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss.backward()
                optimizer.step()
                record=f"{image_type}, Epoch {epoch + 1}/{Epochs}, Loss: {loss.item():.4f}"
                print(record)
                LOSS.append(record)
                count+=1
                if count==all:
                    if epoch %20==0:
                        output_dir = tensor_save+'/'+str(epoch)
                        model.save_pretrained(output_dir, safe_serialization=False)
                        loss_dir=output_dir+'/'+str(epoch+keep_save)+'_epo_notem_noise.csv'
                        df = pd.DataFrame(LOSS, columns=[MODEL_NAME])
                        df.to_csv(loss_dir, index=False)
                

    train(model, datas, epochs=Epochs,lr=learning_rate,tensor_save=tensor_save)
fire.Fire(main)