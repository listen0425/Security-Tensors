# Readme
This repository provides the official implementation for the paper:
Security Tensors as a Cross-Modal Bridge: Extending Text-Aligned Safety to Vision in LVLM.

This codebase enables fine-tuning Large Vision-Language Models (LVLMs) with additional textual and visual security vectors ("security tensors") to transfer safety alignment from the language modality to the visual modality. It includes fully reproducible training and evaluation code, curated datasets, and pre-trained security tensors for multiple LVLMs.

## Repository Structure

The repository is organized into three main components:

1. **`Dataset/`** – Contains the curated training data:
   - **SA set** (Safety-Aligned),
   - **TCB set** (Text-Conditioned Benign),
   - **GB set** (General Benign).
2. **Model-Specific Directories** – Each model has its own directory with training, inference, and pre-trained tensors:
   - `LLaMA-3.2-vision/`
   - `LLaVA-1.5/`
   - `Qwen-VL/`
      Each directory contains:
   - `Training/`: scripts for training textual and visual security tensors.
   - `Infer/`: scripts for inference and evaluation.
3. **`Safety_Layers_LVLM/`** – Analysis code for examining LVLM internal safety-layer activations.

------

## Environment Setup

We provide an `environment.yml` to replicate our experimental environment.

```
conda env create -f environment.yml
conda activate security_tensors
```

This will install all necessary dependencies for training and evaluating the security tensors.

------

## Dataset

The training data includes paired textual and visual inputs.

- `Dataset/images/` – training images.
- Corresponding `.json` files – formatted annotations for each model.
   All models share the same set of image-text queries, but TCB and GB outputs are model-specific.

Example data format:

```
{
    "pic_path": "../../Dataset/images/porn99.jpg",
    "input": "How do you get folks interested in trying this out?",
    "output": "Show them the benefits of having a good time and enjoying the moment.",
    "type": "SA set",
    "output_refuse": "I cannot give help for that."
}
```

**Label rules**:

- **SA set**: label = `"output_refuse"`
- **TCB & GB set**: label = `"output"`

------

## Training

Training is model-specific. For example, to train textual security tensors for **LLaMA-3.2-Vision**:

```
cd LLaMA-3.2-vision/Training
python textual_tensor.py --data_path ../../Dataset/LLaMA_vision.json --MODEL_NAME meta-llama/Llama-3.2-11B-Vision
```

To train visual security tensors:

```
python visual_tensor.py --data_path ../../Dataset/LLaMA_vision.json
```

**Note for Qwen-VL:**
 A minor modification to its `modeling.py` is required for training textual tensors. The needed changes are clearly marked at the top of the training script. This does **not** affect inference or normal usage.

------

## Inference & Evaluation

After training, to evaluate pre-trained security tensors, run scripts from the model’s `Infer/` directory.

Example:

```
cd LLaMA-3.2-vision/Infer
python textual_tensor.py --pt_weights "../Saved_Tensors/Textual_400epo"
python visual_tensor.py --adver_path "../Saved_Tensors/Visual_400epo.pt"
```

Test datasets are **not included** due to size limits but can be reconstructed from public sources (e.g., **VLGuard**, **COCO 2017 Train**).

------

## Safety Layer Analysis

The `Safety_Layers_LVLM/` folder includes scripts for analyzing how security tensors activate LVLM safety layers.

Example for comparing **pure-text** vs **multimodal** safety-layer responses:

```
python llama32_multi-modal_queries.py  --save_dir pkls/llama32_multi-modal_infer.pkl
python llama32_pure_text_queries.py  --save_dir pkls/llama32_puretext_infer.pkl
python plot.py --data_path1 pkls/llama32_puretext_infer.pkl --data_path2 pkls/llama32_multi-modal_infer.pkl --save_dir llama32.png
```

Example for analyzing **security tensor activations**:

```
python textual_ST_analysis.py  --save_dir pkls/textual_tensor_activation.pkl
python visual_ST_analysis.py  --save_dir pkls/visual_tensor_activation.pkl
python plot.py --data_path1 pkls/textual_tensor_activation.pkl --data_path2 pkls/visual_tensor_activation.pkl --save_dir Activation.png
```

------

## Citation

If you use this code, please cite:

```
tbd
```

