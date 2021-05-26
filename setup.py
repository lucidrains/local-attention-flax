from setuptools import setup, find_packages

setup(
    name="local-attention-flax",
    packages=find_packages(),
    version="0.0.1",
    license="MIT",
    description="Local Attention - Flax Module in Jax",
    author="Phil Wang",
    author_email="",
    url="https://github.com/lucidrains/local-attention-flax",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "attention mechanism",
        "jax"
    ],
    install_requires=[
        "einops>=0.3",
        "flax",
        "jax",
        "jaxlib"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
