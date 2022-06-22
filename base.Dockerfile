from pytorch/pytorch

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install \
        wget \
        curl \
        git \
        make \
        sudo

COPY requirements.txt .

RUN conda update conda
RUN conda update conda-build

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

