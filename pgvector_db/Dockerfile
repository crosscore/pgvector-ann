# rag-pgvector/pgvector_db/Dockerfile

FROM postgres:latest

RUN apt-get update && apt-get install -y \
    postgresql-server-dev-all \
    gcc \
    git \
    make \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# タイムゾーンを日本時間（JST）に設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN cd /tmp && \
    git clone --branch v0.7.2 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc && \
    echo "alias ll='ls -alF'" >> ~/.bashrc

RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" > /docker-entrypoint-initdb.d/10-create-extension.sql

# copy custom postgresql.conf
COPY postgresql.conf /etc/postgresql/postgresql.conf

# PostgreSQLの起動コマンドを変更して、カスタム設定ファイルを使用
CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]
