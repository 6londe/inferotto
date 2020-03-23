FROM tensorflow/tensorflow:1.15.2-py3

RUN pip install keras requests sklearn numpy

WORKDIR /app
COPY . .
CMD python infer.py
