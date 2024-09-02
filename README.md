# <img src="icon.png" alt="Icon" style="width:50px; height:50px;"> Law of Vision Representation in MLLMs

[arXiv](https://arxiv.org/abs/2408.16357) / [HuggingFace](https://huggingface.co/papers/2408.16357) / [More Thoughts (Blog in English)](https://huggingface.co/blog/Borise/law-vision-representation-in-mllms) / [More Thoughts (Blog in Chinese)](https://zhuanlan.zhihu.com/p/717297186)

<div align="center">
  <img src="law_gif_fix.gif" alt="Visualization of the law" width="500"/>
</div>




## Updates
- [TODO] Hard working on cleaning the code...but the messy version is here. Please stay tuned and star our repo! 😝
- [2024/09/01] We released the checkpoints of MLLMs on 13 vision representations.
- [2024/08/29] We introduce the Law of Vision Representation in MLLMs and AC Policy.

## Contents
- [Clone This Repository](#clone-this-repository)
- [Train LLaVA with Custom Vision Representation](#train-llava-with-custom-vision-representation)
- [Pretrained Weights](pretrained-weights)
- [Evaluations](evaluations)

## Clone This Repository
```bash
git clone https://github.com/bronyayang/Law_of_Vision_Representation_in_MLLMs.git
cd Law_of_Vision_Representation_in_MLLMs
```

## Train LLaVA with Custom Vision Representation

### 1. Install the LLaVA Environment: Ensure that the environment is compatible with your custom vision module

```Shell
conda create -n ac python=3.10 -y
conda activate ac
pip install --upgrade pip
bash run.sh
```

This training environment has been tested on **CUDA 12.2** and is compatible with all the encoders mentioned in the paper, **except for OpenCLIP** (refer to [environment record](https://github.com/bronyayang/Law_of_Vision_Representation_in_MLLMs/blob/master/version_rec.text) for details on OpenCLIP compatibility).



#### **Important Note:**
To accommodate diffusion model encoders, this environment includes the `diffusers`, `xformers`, and `transformers` packages. However, these packages may conflict with each other. **It is strongly advised to modify [pyproject.toml](https://github.com/bronyayang/Law_of_Vision_Representation_in_MLLMs/blob/master/pyproject.toml) and install only the packages required for your custom vision encoder, rather than all 10 encoders simultaneously.**

### 2. Stage 1 Training

**Prepare LLaVA Stage 1 Data:** Follow the instructions in [LLaVA's tutorial](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#pretrain-feature-alignment) to prepare the data for Stage 1 training.

**Start Training:**
Use the following command to start training:

```Shell
bash llava/scripts/v1_5/train/pretrain.sh
```

However, before running the command, ensure that you modify the following parameters in the script:

- `--data_path`
- `--image_folder`
- `--output_dir`
- `--vision_tower`

**Available Vision Towers:**

- `openai/clip-vit-large-patch14`
- `openai/clip-vit-large-patch14-336`
- `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`
- `google/siglip-base-patch16-224`
- `facebook/dinov2-large`
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2-1`
- `lambdalabs/sd-image-variations-diffusers`
- `stabilityai/stable-diffusion-xl-base-1.0`
- `facebook/DiT-XL-2-512`
- `stabilityai/stable-diffusion-3-medium-diffusers`

**Note:** To combine features from multiple vision towers, use a dot `.` between the names. For example: `openai/clip-vit-large-patch14.facebook/dinov2-large`

Add . between names, such as "openai/clip-vit-large-patch14.facebook/dinov2-large" for feature combination.

### 3. Stage 2 Training
**Prepare LLaVA Stage 2 Data:** Follow the instructions in [LLaVA's tutorial](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning) to prepare the data for Stage 2 training.

**Start Training:**
Use the following command to start training:

```Shell
bash llava/scripts/v1_5/train/pretrain.sh
```

However, before running the command, ensure that you modify the following parameters in the script:

- `--data_path`
- `--image_folder`
- `--output_dir`
- `--vision_tower`
- `--pretrain_mm_mlp_adapter` (checkpoint from Stage 1)

## Pretrained Weights

If you prefer to use the same vision representations that we tested in our paper, we have released [pretrained weights](https://huggingface.co/models?other=arxiv:2408.16357) in Hugging Face for your convenience. This allows you to bypass the steps mentioned above and proceed directly to the next sections.

## Evaluations

Under reconstruction below here...

<!-- ## Acknowledgement -->