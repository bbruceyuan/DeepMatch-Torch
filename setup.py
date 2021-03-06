import setuptools
import deepmatch_torch


with open("README.md", "r") as fh:
    long_description = fh.read()

# 去除了 PyTorch 的依赖，需要用户自己自己下载
REQUIRED_PACKAGES = [
    'tqdm', 'sklearn', 'deepctr-torch', 'tensorflow', 'pytorch-lightning'
]

setuptools.setup(
    name="deepmatch-torch",
    version=deepmatch_torch.__version__,
    author="bbruceyuan",
    author_email="bruceyuan123@gmail.com",
    description="DeepMatch-Torch is a PyTorch Version deep matching model library for recommendations & advertising. It's easy to train models and to export representation vectors for user and item which can be used for ANN search.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bbruceyuan/DeepMatch-Torch",
    download_url='https://github.com/bbruceyuan/DeepMatch-Torch',
    packages=setuptools.find_packages(
        exclude=["tests", "tests.models", "tests.layers"]),
    python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*",  # '>=3.4',  # 3.4.6
    install_requires=REQUIRED_PACKAGES,
    extras_require={

    },
    entry_points={
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="MIT License",
    keywords=['match', "deepmatch_torch", "deepmatch-torch", "deepmatch-pytorch",
              'deep learning', 'torch', 'tensor', 'pytorch', 'deepmatch'],
)