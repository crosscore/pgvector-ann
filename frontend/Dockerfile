# pgvector-ann/frontend/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# タイムゾーンを日本時間（JST）に設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc && \
    echo "alias ll='ls -alF'" >> ~/.bashrc && \
    echo "export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p static templates

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
