import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'torch>=1.1.0', 'tqdm', 'sklearn'
]

setuptools.setup(
    name="deepmatch-torch",
    version="0.0.1",
    author="bbruceyuan",
    author_email="bruceyuan123@gmail.com",
    description="DeepMatch_Torch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bbruceyuan/DeepMatch_Torch",
    download_url='https://github.com/bbruceyuan/DeepMatch_Torch',
    packages=setuptools.find_packages(
        exclude=["tests", "tests.models", "tests.layers"]),
    python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*",  # '>=3.4',  # 3.4.6
    install_requires=REQUIRED_PACKAGES,
    extras_require={

    },
    entry_points={
    },
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0",
    keywords=['match', "deepmatch_torch"
              'deep learning', 'torch', 'tensor', 'pytorch', 'deepmatch'],
)