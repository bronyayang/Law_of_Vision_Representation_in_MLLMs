import os
from setuptools import setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"

setup(
    name="geo-aware",
    author="Junyi Zhang",
    description="Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence",
    python_requires=">=3.8",
    py_modules=[],
    install_requires=[
        "loguru>=0.5.3",
        "faiss-cpu>=1.7.1",
        "matplotlib>=3.4.2",
        "tqdm>=4.61.2",
        "numpy>=1.21.0",
        "gdown>=4.6.0",
        "wandb>=0.16.0",
        f"mask2former @ file://localhost/{os.getcwd()}/third_party/Mask2Former/",
        f"odise @ file://localhost/{os.getcwd()}/third_party/ODISE/"
    ],
    include_package_data=True,
)