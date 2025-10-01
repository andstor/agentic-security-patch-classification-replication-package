# Use Python 3.12 as base image
FROM python:3.12

ENV PYTHONUNBUFFERED=1

# Install system dependencies: JDK 21, git, curl, gcc/g++, bash, php, wget, unzip
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y openjdk-21-jdk git curl gcc g++ bash php wget unzip libnss3 libncurses6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
RUN pip install --upgrade setuptools

# Install Joern in /usr/local/bin
WORKDIR /tmp
RUN wget https://github.com/joernio/joern/releases/download/v4.0.426/joern-install.sh && \
    chmod +x joern-install.sh && \
    ./joern-install.sh --dest /usr/local/bin && \
    rm joern-install.sh

# Set working directory for project
WORKDIR /workspace/

# Copy project files
COPY vul_fix_check.py ./vul_fix_check.py
COPY utils.py ./utils.py
COPY vendor ./vendor