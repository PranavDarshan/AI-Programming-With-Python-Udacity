FROM continuumio/miniconda3
MAINTAINER  PRANAV DARSHAN
ADD /src/environment.yaml /src/environment.yaml
RUN conda env create -f environment.yaml
RUN run_models_batch
