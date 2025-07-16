# setup.py
from setuptools import setup, find_packages

setup(
    name="clip_tuner",
    version="0.1.0",
    description="A thin wrapper for CLIP adaptation",
    packages=find_packages(),  # 自动找到 clip_tuner 包
    install_requires=[
        "torch>=1.10",
        # …你库的依赖…
    ],
    python_requires=">=3.7",
)
