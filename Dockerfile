FROM continuumio/anaconda3:2020.02

RUN pip install keras requests sklearn
RUN pip install tensorflow

WORKDIR /app
COPY . .
CMD python infer.py
