# Use python:3.12 as base image
FROM python:3.12

RUN apt-get update && apt-get install -y vim && rm -rf /var/lib/apt/lists/*

# Install dependencies
ENV PYTHONUNBUFFERED=1
RUN pip install --upgrade pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Set the working directory inside the container
WORKDIR /workspace/

#COPY  .env ./.env
COPY  main.py ./main.py
COPY  --from=rootcontext ./src ./src
