# pgvector-ann/backend/Dockerfile

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y iputils-ping tzdata postgresql-client && \
    rm -rf /var/lib/apt/lists/*

# タイムゾーンを日本時間（JST）に設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc && \
    echo "alias ll='ls -alF'" >> ~/.bashrc && \
    echo "export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# FastAPIアプリケーションを起動。
# "main:app": main.py (app = FastAPI())
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
