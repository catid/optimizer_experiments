conda create -n gpt python=3.10
conda activate gpt

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install -U "ray[data,train,tune,serve]==2.53.0"

pip install wandb transformers==4.47.1 tqdm setuptools accelerate==1.12.0 scipy loguru datasets==4.0.0