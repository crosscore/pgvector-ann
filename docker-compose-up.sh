#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    docker compose -f docker-compose.macos.yml up --build
else
    docker compose -f docker-compose.linux.yml up --build
fi
