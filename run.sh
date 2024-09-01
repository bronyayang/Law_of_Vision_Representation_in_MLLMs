pip3 install -e .
pip install -e ".[train]"
pip3 install flash-attn --no-build-isolation
pip install wandb
pip3 install transformers==4.38.2
pip3 install tokenizers==0.15.2
pip3 install accelerate==0.27.2
pip3 install deepspeed==0.12.6
