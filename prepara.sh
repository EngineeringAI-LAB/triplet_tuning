conda create -n t4 python==3.10 -y
conda activate t4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd /mnt/afse1/zhenhong/downloads
pip install spacy==3.8.0
pip install en_core_web_trf-3.8.0-py3-none-any.whl
pip install en_core_web_sm-3.8.0-py3-none-any.whl

pip install huggingface-hub==0.23.4 gradio==4.0.1 diffusers==0.28.0 transformers==4.37.2 accelerate==0.23.0 albumentations wandb  seaborn  datasets nltk 

# https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.0.0/en_core_web_trf-3.0.0-py3-none-any.whl
# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# https://github.com/nltk/nltk_data