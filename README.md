## T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Generation

## Maintained Repository

This repository is a **maintained mirror** of the official implementation released at:

👉 https://github.com/Tencent/Triplet_Tuning

### Purpose

The purpose of this repository is to support **reproducibility**, **continued maintenance**, and **future extensions** of the Triplet Tuning framework, including benchmark updates and additional experimental settings.

The original implementation and its initial release are preserved in the official repository.  
This repository tracks the official codebase while providing a stable location for ongoing updates.


> #### Authors &emsp;&emsp; Zhenhong Sun, Yifu Wang, Yonhon Ng, Yongzhi Xu, Daoyi Dong, Hongdong Li, Pan Ji 

> #### Abstract
Scene generation is crucial to many computer graphics applications. Recent advances in generative AI have streamlined sketch-to-image workflows, easing the workload for artists and designers in creating scene concept art. However, these methods often struggle for complex scenes with multiple detailed objects, sometimes missing small or uncommon instances. In this paper, we propose a Training-free Triplet Tuning for Sketch-to-Scene (T3-S2S) generation after reviewing the entire cross-attention mechanism. This scheme revitalizes the existing ControlNet model, enabling effective handling of multi-instance generations, involving prompt balance, characteristics prominence, and dense tuning. Specifically, this approach enhances keyword representation via the prompt balance module, reducing the risk of missing critical instances. It also includes a characteristics prominence module that highlights TopK indices in each channel, ensuring essential features are better represented based on token sketches. Additionally, it employs dense tuning to refine contour details in the attention map, compensating for instance-related regions. Experiments validate that our triplet tuning approach substantially improves the performance of existing sketch-to-image models. It consistently generates detailed, multi-instance 2D images, closely adhering to the input prompts and enhancing visual quality in complex multi-instance scenes.


> #### Environment and Models
```shell
## Environment Setup

This project uses **Python 3.10** and is tested with **CUDA 12.1**.

### 1. Create and activate conda environment

conda create -n t3 python=3.10 -y
conda activate t3

### 2. Install PyTorch (CUDA 12.1)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

### 3. Install spaCy and language models

pip install spacy==3.8.0

# Install spaCy English models (download the .whl files first)
pip install en_core_web_trf-3.8.0-py3-none-any.whl
pip install en_core_web_sm-3.8.0-py3-none-any.whl

# Model download references:
# https://github.com/explosion/spacy-models/releases
# https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.0.0/en_core_web_trf-3.0.0-py3-none-any.whl
# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

### 4. Install core dependencies

pip install huggingface-hub==0.23.4 gradio==4.0.1 diffusers==0.28.0 \
            transformers==4.37.2 accelerate==0.23.0 albumentations \
            wandb seaborn datasets nltk

### 5. (Optional) Download NLTK data

python -m nltk.downloader all

# NLTK data reference:
# https://github.com/nltk/nltk_data

git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
git clone https://huggingface.co/xinsir/controlnet-union-sdxl-1.0
```

### How to launch a web interface

- Run the Gradio app.
```shell
python gradio_triplet.py
python gradio_dual.py
# use the examples for the evaluation.
# for the complex colored cases, please adjust hyperparameters.
```

----


#### BibTeX
```
@article{sun2024T3S2S,
      title={T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Generation}, 
      author={Zhenhong Sun and Yifu Wang and Yonhon Ng and Yunfei Duan and Daoyi Dong and Hongdong Li and Pan Ji},
      year={2024},
      eprint={2412.13486},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.13486}, 
}
``` 
or
```
@journal{sun2025t3s2s,
  title   = {{T}$^3$-S2S: Training-free Triplet Tuning for Sketch-to-Scene Generation},
  author  = {Sun, Zhenhong and Wang, Yifu and Ng, Yonhon and Xu, Yongzhi and Dong, Daoyi and Li, Hongdong and Ji, Pan},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=lyn2BgKQ8F}
}
```
---

#### Acknowledgment
The demo was developed referencing this [source code](https://github.com/naver-ai/DenseDiffusion). Thanks for the inspiring work! 🙏 

