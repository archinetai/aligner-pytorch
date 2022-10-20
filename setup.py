from Cython.Build import cythonize
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="aligner-pytorch",
    packages=find_packages(exclude=[]),
    version="0.0.2",
    license="MIT",
    description="Aligner - PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Flavio Schneider",
    author_email="archinetai@protonmail.com",
    url="https://github.com/archinetai/audio-diffusion-pytorch",
    keywords=["artificial intelligence", "deep learning", "TTS", "alignment"],
    setup_requires=["cython"],
    install_requires=[
        "torch>=1.6",
        "data-science-types>=0.2",
        "einops>=0.4",
        "einops-exts>=0.0.3",
    ],
    ext_modules=cythonize(["aligner_pytorch/*.pyx", "*.pyx"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
