from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eva",
    version="0.1.0",
    author="Yufan Liu",
    description="Eva, a multimodal self-supervised foundation model for spatial proteomics and histology data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "einops>=0.6.0",
        "huggingface-hub>=0.16.0",
        "numpy>=1.21.0",
    ],
)

