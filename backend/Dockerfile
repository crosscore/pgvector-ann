# rag-pgvecotr/backend/Dockerfile

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y iputils-ping tzdata && \
    rm -rf /var/lib/apt/lists/*

# タイムゾーンを日本時間（JST）に設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc && \
    echo "alias ll='ls -alF'" >> ~/.bashrc

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]
