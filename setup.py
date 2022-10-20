import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

ext_modules = cythonize(
    ["aligner_pytorch/mas_c.pyx"],
    compiler_directives={"language_level": "3"},
)

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
    install_requires=[
        "numpy",
        "torch>=1.6",
        "data-science-types>=0.2",
        "einops>=0.4",
        "einops-exts>=0.0.3",
    ],
    include_dirs=[numpy.get_include(), "monotonic_align"],
    ext_modules=cythonize(ext_modules),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
