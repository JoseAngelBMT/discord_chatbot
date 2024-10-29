FROM python:3.12-slim

COPY src/ /myapp/src/
RUN ls -la /myapp/*
COPY requirements.txt ./
COPY requirements_torch.txt ./

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    ninja-build \
    libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip3 install -r requirements_torch.txt --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

WORKDIR /myapp
ENV PYTHONPATH=/myapp

EXPOSE 5000

CMD ["python", "src/app/main.py"]
