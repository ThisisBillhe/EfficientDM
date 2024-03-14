# EFFICIENTDM: EFFICIENT QUANTIZATION-AWARE FINE-TUNING OF LOW-BIT DIFFUSION MODELS

This repository provides the implementation for our paper "EFFICIENTDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models". Our approach introduces a novel method for fine-tuning low-bit diffusion models, focusing on efficiency and performance.

[arXiv](https://arxiv.org/abs/2310.03270) | [BibTeX](#bibtex)

## Getting Started

Follow the step-by-step tutorial to set up EFFICIENTDM.

### Step 1: Setup
Create a virtual environment and install dependencies as specified by LDM.

### Step 2: Download Pretrained Models
Download the pretrained models provided by LDM.
```shell
mkdir -p models/ldm/cin256-v2/
wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
```

### Step 3: Collect Input Data for Calibration
Gather input data required for model calibration. Remember to modified the ldm/models/diffusion/ddpm.py as indicated in the quant_scripts/collect_input_4_calib.py.
```python
python3 quant_scripts/collect_input_4_calib.py
```
### Step 4: Quantize and Calibrate the Model
We just apply a naive quantization method for model calibration because we will fine-tune it afterwards.
```python
python3 quant_scripts/quantize_ldm_naive.py
```
### Step 5: Convert Quantized Model
Convert the fake quantized model to a packed integer model. Notice the significant reduction in model size.
```python
python3 quant_scripts/save_naive_2_intmodel.py
```

### Step 6: Fine-Tune with EfficientDM
```python
python3 quant_scripts/train_efficientdm.py

```

### Step 7 (Optional): Downsample TALSQ Parameters
Optionally, downsample the Temporal Activation LSQ parameters for sampling with fewer steps.
```python
python3 quant_scripts/downsample_talsq_ckpt.py

```

### Step 8: Sample with the EfficientDM Model
```python
python3 quant_scripts/sample_lora_intmodel.py

```
## Fine-tuned EfficientDM Weights

| Model | Dataset | Link |
| ------- | ------- |------- |
|  LDM-4  | ImageNet|https://drive.google.com/file/d/1xSGY5lXnBhXK9beq3j1NUXkFRr76mDLO/view|

## BibTeX
If you find this work useful for your research, please consider citing:
```
@inproceedings{he2024efficientdm,
  title={EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models},
  author={He, Yefei and Liu, Jing and Wu, Weijia and Zhou, Hong and Zhuang, Bohan},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
