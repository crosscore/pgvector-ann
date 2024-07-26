#!/bin/bash

if docker compose ps | grep "Up"; then
    echo "Stopping Docker containers..."
    docker compose down
else
    echo "Starting Docker containers..."
    docker compose up --build
fi
