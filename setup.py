from setuptools import find_packages, setup
from Cython.Build import cythonize

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="aligner-pytorch",
    version="0.0.19",
    packages=find_packages(),
    license="MIT",
    description="Aligner - PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Flavio Schneider",
    author_email="archinetai@protonmail.com",
    url="https://github.com/archinetai/audio-diffusion-pytorch",
    keywords=["artificial intelligence", "deep learning", "TTS", "alignment"],
    install_requires=[
        "torch>=1.6",
        "data-science-types>=0.2",
        "einops>=0.4",
        "einops-exts>=0.0.3",
    ],
    setup_requires=["cython"],
    include_dirs=["aligner_pytorch"],
    ext_modules=cythonize(["aligner_pytorch/mas_c.pyx"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
