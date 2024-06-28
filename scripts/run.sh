cd /opt/tiger/LLaVA1.5
pip3 install -e .
pip install -e ".[train]"
pip3 install ninja 
pip3 install flash-attn --no-build-isolation
pip3 install -U byted-wandb bytedfeather -i "https://bytedpypi.byted.org/simple"
pip3 install transformers==4.38.2
pip3 install tokenizers==0.15.2
pip3 install accelerate==0.27.2
pip3 install deepspeed==0.12.6
