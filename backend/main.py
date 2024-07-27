# pgvector-ann/backend/main.py
import asyncio
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import psycopg2
from openai import OpenAI
from starlette.websockets import WebSocketDisconnect
import logging
from contextlib import contextmanager
from pypdf import PdfReader, PdfWriter
import time
import os
import pandas as pd
from utils.docker_stats_csv import get_container_memory_stats, save_memory_stats_to_csv
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"Application initialized with INDEX_TYPE: {INDEX_TYPE}, "
            f"IVFFLAT_PROBES: {IVFFLAT_PROBES}, HNSW_EF_SEARCH: {HNSW_EF_SEARCH}")

client = OpenAI(api_key=OPENAI_API_KEY)

@contextmanager
def get_db_connection():
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            dbname=PGVECTOR_DB_NAME,
            user=PGVECTOR_DB_USER,
            password=PGVECTOR_DB_PASSWORD,
            host=PGVECTOR_DB_HOST,
            port=PGVECTOR_DB_PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()
        if INDEX_TYPE == "ivfflat":
            cursor.execute(f"SET ivfflat.probes = {IVFFLAT_PROBES};")
            logger.info(f"Set ivfflat.probes to {IVFFLAT_PROBES}")
        elif INDEX_TYPE == "hnsw":
            cursor.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH};")
            logger.info(f"Set hnsw.ef_search to {HNSW_EF_SEARCH}")
        conn.autocommit = False
        yield conn, cursor
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_search_query(index_type):
    vector_type = "halfvec(3072)" if index_type in ["hnsw", "ivfflat"] else "vector(3072)"
    return f"""
    SELECT file_name, document_page, chunk_no, chunk_text,
            (chunk_vector::{vector_type} <#> %s::{vector_type}) AS distance
    FROM document_vectors
    ORDER BY distance ASC
    LIMIT %s;
    """

async def collect_memory_stats(container_name, duration, interval=0.1):
    start_time = time.time()
    stats_list = []
    while time.time() - start_time < duration:
        stats = get_container_memory_stats(container_name)
        if stats:
            stats_list.append(stats)
        await asyncio.sleep(interval)
    return stats_list

def save_memory_stats_with_extra_info(stats_list, filename, index_type, search_time, keyword):
    df_list = []
    for stats in stats_list:
        df = pd.DataFrame([stats])
        df['index_type'] = index_type if index_type in ['hnsw', 'ivfflat'] else 'None'
        df['search_time'] = search_time
        df['keyword'] = keyword
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # Reorder columns
    columns = ['index'] + ['index_type', 'search_time', 'keyword'] + [col for col in df.columns if col not in ['index', 'index_type', 'search_time', 'keyword']]
    df = df[columns]

    df.to_csv(filename, index_label='index')
    logger.info(f"Memory stats saved to {filename}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            question, top_n = data["question"], data.get("top_n", 5)

            question_vector = client.embeddings.create(
                input=question,
                model="text-embedding-3-large"
            ).data[0].embedding

            container_name = os.getenv('POSTGRES_CONTAINER_NAME', 'pgvector_db')
            logger.info(f"Attempting to get stats for container: {container_name}")

            # 検索前のメモリ使用量を取得
            before_search_stats = await collect_memory_stats(container_name, duration=1)

            start_time = time.time()
            try:
                with get_db_connection() as (conn, cursor):
                    cursor.execute(get_search_query(INDEX_TYPE), (question_vector, top_n))
                    results = cursor.fetchall()
                    conn.commit()

                    # 検索中のメモリ使用量を取得
                    during_search_stats = await collect_memory_stats(container_name, duration=5)

                search_time = time.time() - start_time

                # 検索後のメモリ使用量を取得
                after_search_stats = await collect_memory_stats(container_name, duration=1)

                formatted_results = [
                    {
                        "file_name": file_name,
                        "page": document_page,
                        "chunk_no": chunk_no,
                        "chunk_text": chunk_text,
                        "distance": float(distance),
                        "link_text": f"{file_name}, p.{document_page}",
                        "link": f"/pdf/{file_name}?page={document_page}",
                    }
                    for file_name, document_page, chunk_no, chunk_text, distance in results
                ]

                save_memory_stats_with_extra_info(before_search_stats, "./data/csv/before_search.csv", INDEX_TYPE, search_time, question)
                save_memory_stats_with_extra_info(during_search_stats, "./data/csv/during_search.csv", INDEX_TYPE, search_time, question)
                save_memory_stats_with_extra_info(after_search_stats, "./data/csv/after_search.csv", INDEX_TYPE, search_time, question)

                response_data = {
                    "results": formatted_results,
                    "search_time": search_time,
                }
                await websocket.send_json(response_data)
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=8001)
