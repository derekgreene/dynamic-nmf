# Build an image for the preprocessing of
# congress data for JMParelman project
# https://github.com/jmparelman/dynamic-nmf

# Pull base image
FROM python:3.7-slim-buster

MAINTAINER ASC "mbod@asc.upenn.edu jmparelman@gmail.com"

# setup working directory
RUN mkdir -p /data/input /data/output /working

# install OS modules - especially Python
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        build-essential \
        python-dev \
        python3-pip \
    && pip3 install -U pip


ENV SPACY_VERSION 3.0.5

# install Python packages using pip install
RUN \
    pip3 install -U pandas \
         tqdm \
         gensim \
         sklearn \
    && pip3 install -U spacy==${SPACY_VERSION} \
    && python3 -m spacy download en_core_web_sm


WORKDIR /working

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
