---
layout: post
title:  "A Basic Environment Used for Deep Learnig and Computer Vision"
date:   2024-08-11 16:18:12 -0700
categories: Coding
---

This is a basic conda environment used by me to run deep learning and computer vision experiments.

## Create a Conda Virtual Environment

To create an environment named `ENV_NAME` with python 3.10, using the channel conda-forge and a list of packages:

```bash
conda create -y --name py310_pt220 python==3.10
```

Use this command to remove the environment:
```bash
conda remove -n ENV_NAME --all
```

## Common Packages

```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install -c conda-forge nodejs
conda install -c conda-forge ipywidgets
conda install -c conda-forge jupyterlab
conda install -c conda-forge pillow
```

To automate a backup of the current environment run:
```bash
conda env export --no-builds | grep -v "prefix" > environment.yml
```
Then you can get a `.yml` file as:

```yml
name: py310_pt220
channels:
  - pytorch
  - nvidia
  - nvidia/label/cuda-12.1.0
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1
  - _openmp_mutex=4.5
  - _sysroot_linux-64_curr_repodata_hack=3
  - anyio=4.4.0
  - argon2-cffi=23.1.0
  - argon2-cffi-bindings=21.2.0
  - arrow=1.3.0
  - asttokens=2.4.1
  - async-lru=2.0.4
  - attrs=24.2.0
  - babel=2.14.0
  - beautifulsoup4=4.12.3
  - binutils=2.38
  - binutils_impl_linux-64=2.38
  - binutils_linux-64=2.38.0
  - blas=1.0
  - bleach=6.1.0
  - brotli-python=1.0.9
  - bzip2=1.0.8
  - ca-certificates=2024.8.30
  - cached-property=1.5.2
  - cached_property=1.5.2
  - certifi=2024.8.30
  - cffi=1.17.1
  - charset-normalizer=3.3.2
  - comm=0.2.2
  - cuda-cccl=12.1.55
  - cuda-command-line-tools=12.1.0
  - cuda-compiler=12.4.1
  - cuda-cudart=12.1.105
  - cuda-cudart-dev=12.1.55
  - cuda-cudart-static=12.1.55
  - cuda-cuobjdump=12.4.127
  - cuda-cupti=12.1.105
  - cuda-cupti-static=12.1.62
  - cuda-cuxxfilt=12.4.127
  - cuda-documentation=12.1.55
  - cuda-driver-dev=12.1.55
  - cuda-gdb=12.1.55
  - cuda-libraries=12.1.0
  - cuda-libraries-dev=12.1.0
  - cuda-libraries-static=12.1.0
  - cuda-nsight=12.1.55
  - cuda-nsight-compute=12.1.0
  - cuda-nvcc=12.4.131
  - cuda-nvdisasm=12.1.55
  - cuda-nvml-dev=12.1.55
  - cuda-nvprof=12.1.55
  - cuda-nvprune=12.4.127
  - cuda-nvrtc=12.1.105
  - cuda-nvrtc-dev=12.1.55
  - cuda-nvrtc-static=12.1.55
  - cuda-nvtx=12.1.105
  - cuda-nvvp=12.1.55
  - cuda-opencl=12.6.68
  - cuda-opencl-dev=12.1.56
  - cuda-profiler-api=12.1.55
  - cuda-runtime=12.1.0
  - cuda-sanitizer-api=12.1.55
  - cuda-toolkit=12.1.0
  - cuda-tools=12.1.0
  - cuda-version=12.6
  - cuda-visual-tools=12.1.0
  - debugpy=1.8.5
  - decorator=5.1.1
  - defusedxml=0.7.1
  - entrypoints=0.4
  - exceptiongroup=1.2.2
  - executing=2.1.0
  - ffmpeg=4.4.0
  - filelock=3.13.1
  - fqdn=1.5.1
  - freetype=2.12.1
  - gcc_impl_linux-64=11.2.0
  - gcc_linux-64=11.2.0
  - gds-tools=1.6.0.25
  - gmp=6.2.1
  - gmpy2=2.1.2
  - gnutls=3.6.15
  - gxx_impl_linux-64=11.2.0
  - gxx_linux-64=11.2.0
  - h11=0.14.0
  - h2=4.1.0
  - hpack=4.0.0
  - httpcore=1.0.5
  - httpx=0.27.2
  - hyperframe=6.0.1
  - icu=75.1
  - idna=3.7
  - importlib-metadata=8.5.0
  - importlib_metadata=8.5.0
  - importlib_resources=6.4.5
  - intel-openmp=2021.4.0
  - ipykernel=6.29.5
  - ipython=8.27.0
  - ipywidgets=8.1.5
  - isoduration=20.11.0
  - jedi=0.19.1
  - jinja2=3.1.4
  - jpeg=9e
  - json5=0.9.25
  - jsonpointer=3.0.0
  - jsonschema=4.23.0
  - jsonschema-specifications=2023.12.1
  - jsonschema-with-format-nongpl=4.23.0
  - jupyter-lsp=2.2.5
  - jupyter_client=8.6.2
  - jupyter_core=5.7.2
  - jupyter_events=0.10.0
  - jupyter_server=2.14.2
  - jupyter_server_terminals=0.5.3
  - jupyterlab=4.2.5
  - jupyterlab_pygments=0.3.0
  - jupyterlab_server=2.27.3
  - jupyterlab_widgets=3.0.13
  - kernel-headers_linux-64=3.10.0
  - lame=3.100
  - lcms2=2.15
  - ld_impl_linux-64=2.38
  - lerc=4.0.0
  - libcublas=12.1.0.26
  - libcublas-dev=12.1.0.26
  - libcublas-static=12.1.0.26
  - libcufft=11.0.2.4
  - libcufft-dev=11.0.2.4
  - libcufft-static=11.0.2.4
  - libcufile=1.11.1.6
  - libcufile-dev=1.6.0.25
  - libcufile-static=1.6.0.25
  - libcurand=10.3.7.68
  - libcurand-dev=10.3.2.56
  - libcurand-static=10.3.2.56
  - libcusolver=11.4.4.55
  - libcusolver-dev=11.4.4.55
  - libcusolver-static=11.4.4.55
  - libcusparse=12.0.2.55
  - libcusparse-dev=12.0.2.55
  - libcusparse-static=12.0.2.55
  - libdeflate=1.17
  - libffi=3.4.4
  - libgcc=14.1.0
  - libgcc-devel_linux-64=11.2.0
  - libgcc-ng=14.1.0
  - libgomp=14.1.0
  - libiconv=1.16
  - libidn2=2.3.4
  - libjpeg-turbo=2.0.0
  - libnpp=12.0.2.50
  - libnpp-dev=12.0.2.50
  - libnpp-static=12.0.2.50
  - libnsl=2.0.1
  - libnvjitlink=12.1.105
  - libnvjitlink-dev=12.1.55
  - libnvjpeg=12.1.1.14
  - libnvjpeg-dev=12.1.0.39
  - libnvjpeg-static=12.1.0.39
  - libnvvm-samples=12.1.55
  - libpng=1.6.44
  - libsodium=1.0.18
  - libsqlite=3.45.2
  - libstdcxx=14.1.0
  - libstdcxx-devel_linux-64=11.2.0
  - libstdcxx-ng=14.1.0
  - libtasn1=4.19.0
  - libtiff=4.5.0
  - libunistring=0.9.10
  - libuuid=2.38.1
  - libuv=1.48.0
  - libvpx=1.11.0
  - libwebp-base=1.3.2
  - libxcb=1.13
  - libxcrypt=4.4.36
  - libzlib=1.3.1
  - llvm-openmp=15.0.7
  - lz4-c=1.9.4
  - markupsafe=2.1.3
  - matplotlib-inline=0.1.7
  - mistune=3.0.2
  - mkl=2021.4.0
  - mkl-service=2.4.0
  - mkl_fft=1.3.1
  - mkl_random=1.2.2
  - mpc=1.1.0
  - mpfr=4.0.2
  - mpmath=1.3.0
  - nbclient=0.10.0
  - nbconvert-core=7.16.4
  - nbformat=5.10.4
  - ncurses=6.4
  - nest-asyncio=1.6.0
  - nettle=3.7.3
  - networkx=3.2.1
  - nodejs=22.8.0
  - notebook-shim=0.2.4
  - nsight-compute=2023.1.0.15
  - numpy=1.24.3
  - numpy-base=1.24.3
  - openh264=2.1.1
  - openjpeg=2.5.0
  - openssl=3.3.2
  - overrides=7.7.0
  - packaging=24.1
  - pandocfilters=1.5.0
  - parso=0.8.4
  - pexpect=4.9.0
  - pickleshare=0.7.5
  - pillow=9.4.0
  - pip=24.2
  - pkgutil-resolve-name=1.3.10
  - platformdirs=4.3.3
  - prometheus_client=0.20.0
  - prompt-toolkit=3.0.47
  - psutil=6.0.0
  - pthread-stubs=0.4
  - ptyprocess=0.7.0
  - pure_eval=0.2.3
  - pycparser=2.22
  - pygments=2.18.0
  - pysocks=1.7.1
  - python=3.10.13
  - python-dateutil=2.9.0
  - python-fastjsonschema=2.20.0
  - python-json-logger=2.0.7
  - python_abi=3.10
  - pytorch=2.2.0
  - pytorch-cuda=12.1
  - pytorch-mutex=1.0
  - pytz=2024.2
  - pyyaml=6.0.1
  - pyzmq=26.2.0
  - readline=8.2
  - referencing=0.35.1
  - requests=2.32.3
  - rfc3339-validator=0.1.4
  - rfc3986-validator=0.1.1
  - rpds-py=0.20.0
  - send2trash=1.8.3
  - setuptools=72.1.0
  - six=1.16.0
  - sniffio=1.3.1
  - soupsieve=2.5
  - sqlite=3.45.2
  - stack_data=0.6.2
  - sympy=1.13.2
  - sysroot_linux-64=2.17
  - tbb=2021.8.0
  - terminado=0.18.1
  - tinycss2=1.3.0
  - tk=8.6.13
  - tomli=2.0.1
  - torchaudio=2.2.0
  - torchtriton=2.2.0
  - torchvision=0.17.0
  - tornado=6.4.1
  - traitlets=5.14.3
  - types-python-dateutil=2.9.0.20240906
  - typing-extensions=4.11.0
  - typing_extensions=4.11.0
  - typing_utils=0.1.0
  - uri-template=1.3.0
  - urllib3=2.2.2
  - wcwidth=0.2.13
  - webcolors=24.8.0
  - webencodings=0.5.1
  - websocket-client=1.8.0
  - wheel=0.44.0
  - widgetsnbextension=4.0.13
  - x264=1!161.3030
  - xorg-libxau=1.0.11
  - xorg-libxdmcp=1.1.3
  - xz=5.4.6
  - yaml=0.2.5
  - zeromq=4.3.5
  - zipp=3.20.2
  - zlib=1.3.1
  - zstd=1.5.6
  - pip:
      - absl-py==2.1.0
      - aiohappyeyeballs==2.4.0
      - aiohttp==3.10.5
      - aiosignal==1.3.1
      - albucore==0.0.15
      - albumentations==1.4.15
      - annotated-types==0.7.0
      - antlr4-python3-runtime==4.9.3
      - async-timeout==4.0.3
      - contourpy==1.3.0
      - cycler==0.12.1
      - cython==3.0.11
      - datasets==3.0.0
      - diffusers==0.30.2
      - dill==0.3.8
      - einops==0.8.0
      - eval-type-backport==0.2.0
      - fonttools==4.53.1
      - frozenlist==1.4.1
      - fsspec==2024.6.1
      - future==1.0.0
      - grpcio==1.66.1
      - huggingface-hub==0.24.7
      - imageio==2.35.1
      - imageio-ffmpeg==0.5.1
      - invisible-watermark==0.2.0
      - ipdb==0.13.13
      - joblib==1.4.2
      - kiwisolver==1.4.7
      - kornia==0.7.3
      - kornia-rs==0.1.5
      - lazy-loader==0.4
      - lightning-utilities==0.11.7
      - loguru==0.7.2
      - markdown==3.7
      - matplotlib==3.9.2
      - multidict==6.1.0
      - multiprocess==0.70.16
      - natsort==8.4.0
      - nose==1.3.7
      - omegaconf==2.3.0
      - opencv-python==4.10.0.84
      - opencv-python-headless==4.10.0.84
      - pandas==2.2.2
      - path==17.0.0
      - protobuf==5.28.1
      - pudb==2024.1.2
      - pyarrow==17.0.0
      - pydantic==2.9.1
      - pydantic-core==2.23.3
      - pyparsing==3.1.4
      - python-graphviz==0.20.3
      - pytorch-lightning==2.4.0
      - pywavelets==1.7.0
      - regex==2024.9.11
      - safetensors==0.4.5
      - scikit-image==0.24.0
      - scikit-learn==1.5.2
      - scipy==1.14.1
      - tensorboard==2.17.1
      - tensorboard-data-server==0.7.2
      - test-tube==0.7.5
      - threadpoolctl==3.5.0
      - tifffile==2024.8.30
      - timm==1.0.9
      - tokenizers==0.19.1
      - torch-fidelity==0.3.0
      - torchmetrics==1.4.2
      - torchviz==0.0.2
      - tqdm==4.66.5
      - transformers==4.44.2
      - tzdata==2024.1
      - urwid==2.6.15
      - urwid-readline==0.14
      - werkzeug==3.0.4
      - xxhash==3.5.0
      - yarl==1.11.1
```
