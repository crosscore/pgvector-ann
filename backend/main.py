# pgvector-ann/backend/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from starlette.websockets import WebSocketDisconnect
import logging
import time
import os
from utils.docker_stats_csv import save_memory_stats_with_extra_info, collect_memory_stats
from utils.db_utils import get_db_connection, get_search_query
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

            before_search_stats = await collect_memory_stats(container_name, duration=1)

            start_time = time.time()
            try:
                with get_db_connection() as (conn, cursor):
                    cursor.execute(get_search_query(INDEX_TYPE), (question_vector, top_n))
                    results = cursor.fetchall()
                    conn.commit()

                    during_search_stats = await collect_memory_stats(container_name, duration=5)

                search_time = time.time() - start_time

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

                save_memory_stats_with_extra_info(before_search_stats, os.path.join(CSV_OUTPUT_DIR ,'before_search.csv'), INDEX_TYPE, search_time, question)
                save_memory_stats_with_extra_info(during_search_stats, os.path.join(CSV_OUTPUT_DIR, 'during_search.csv'), INDEX_TYPE, search_time, question)
                save_memory_stats_with_extra_info(after_search_stats, os.path.join(CSV_OUTPUT_DIR, 'after_search.csv'), INDEX_TYPE, search_time, question)

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
