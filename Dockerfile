FROM nvidia/cuda:11.6.1-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update \
    && apt-get install -y make gcc git wget curl \
        zlib1g zlib1g-dev python3-pip libssl-dev \
        bzip2 libbz2-dev sqlite3 libsqlite3-dev \
        libreadline-dev libffi-dev liblzma-dev \
        graphviz graphviz-dev scons build-essential \
        python3-dev cmake ffmpeg libsm6 libxext6

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID app \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID app

USER app

ENV HOME /home/app

WORKDIR $HOME

RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

ENV PYTHON_VERSION 3.8.3

RUN pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pip install --upgrade setuptools pip

ENV PATH $PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH "${PATH}:${HOME}/.poetry/bin"

ENV PYTHONPATH /app
ENV PYTHONDONTWRITEBYTECODE 1

COPY requirements.txt.lock requirements.txt.lock
RUN pip install -r requirements.txt.lock

# RUN git clone https://github.com/NVIDIA/apex; \
#     cd apex; \
#     sed -i 's/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/print(42)/g' setup.py; \
#     pip --no-cache-dir install --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# RUN git clone --recurse-submodules https://github.com/dmlc/dgl.git; \
#     cd dgl; \
#     mkdir build; \
#     cd build; \
#     cmake -DUSE_CUDA=ON ..; \
#     make -j4; \
#     cd ../python; \
#     python setup.py install

WORKDIR /app

RUN pip install opencv-python scikit-image
