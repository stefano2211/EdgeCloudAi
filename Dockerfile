FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libcurl4-openssl-dev \ 
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ollama start & \
    sleep 5 && \
    ollama serve && \
    kill $(pgrep ollama)

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["sh", "-c", "ollama serve & uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload"]