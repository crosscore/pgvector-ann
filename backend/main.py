# pgvector-ann/backend/main.py
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import psycopg2
from openai import OpenAI
from starlette.websockets import WebSocketDisconnect
import logging
from contextlib import contextmanager
from pypdf import PdfReader, PdfWriter
from io import BytesIO
import time
import docker
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
docker_client = docker.from_env()

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

def get_container_memory_stats(container_name):
    try:
        container = docker_client.containers.get(container_name)
        stats = container.stats(stream=False)
        memory_stats = stats['memory_stats']
        return {
            'total_memory_usage': memory_stats['usage'],
            'rss': memory_stats['stats']['rss']
        }
    except docker.errors.NotFound:
        logger.error(f"Container {container_name} not found")
        return None
    except Exception as e:
        logger.error(f"Error getting container stats: {str(e)}")
        return None

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

            # メモリ使用量を取得（検索前）
            pre_search_memory = get_container_memory_stats('your_postgres_container_name')

            start_time = time.time()
            try:
                with get_db_connection() as (conn, cursor):
                    cursor.execute(get_search_query(INDEX_TYPE), (question_vector, top_n))
                    results = cursor.fetchall()
                    conn.commit()

                search_time = time.time() - start_time

                # メモリ使用量を取得（検索後）
                post_search_memory = get_container_memory_stats('your_postgres_container_name')

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

                # メモリ使用量の変化を計算（変化がない場合も0として報告）
                memory_change = {
                    'total_memory_usage': 0,
                    'rss': 0
                }
                if pre_search_memory and post_search_memory:
                    memory_change['total_memory_usage'] = post_search_memory['total_memory_usage'] - pre_search_memory['total_memory_usage']
                    memory_change['rss'] = post_search_memory['rss'] - pre_search_memory['rss']

                response_data = {
                    "results": formatted_results,
                    "search_time": search_time,
                    "memory_usage_change": memory_change
                }
                await websocket.send_json(response_data)
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

@app.get("/pdf/{file_name}")
async def get_pdf(file_name: str, page: int):
    pdf_path = os.path.join(PDF_INPUT_DIR, file_name)

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {file_name}")

    try:
        reader = PdfReader(pdf_path)

        if not (0 <= page < len(reader.pages)):
            raise HTTPException(status_code=400, detail=f"Invalid page number: {page}. Valid range is 0-{len(reader.pages)-1}")

        writer = PdfWriter()
        writer.add_page(reader.pages[page])
        output_filename = f"{file_name.rsplit('.', 1)[0]}_page_{page}.pdf"

        buffer = BytesIO()
        writer.write(buffer)
        buffer.seek(0)

        headers = {
            "Content-Disposition": f"inline; filename={output_filename}"
        }
        return StreamingResponse(buffer, media_type="application/pdf", headers=headers)

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing PDF")

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=8001)
