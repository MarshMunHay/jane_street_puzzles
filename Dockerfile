FROM python:3.10.0-buster as python
WORKDIR /project

RUN apt-get update && apt-get -y install cmake protobuf-compiler

COPY ./requirements.txt /project/
RUN pip3 install -r requirements.txt
