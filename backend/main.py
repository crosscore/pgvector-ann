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
import psutil
import statistics
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

# Dockerクライアントの初期化
docker_client = docker.from_env()

def get_pgvector_db_memory_usage():
    try:
        container = docker_client.containers.get('pgvector-ann_pgvector_db_1')
        stats = container.stats(stream=False)
        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB単位に変換
        return memory_usage
    except Exception as e:
        logger.error(f"Error getting pgvector_db memory usage: {str(e)}")
        return None

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
        cursor = conn.cursor()
        if INDEX_TYPE == "ivfflat":
            cursor.execute(f"SET ivfflat.probes = {IVFFLAT_PROBES};")
            logger.info(f"Set ivfflat.probes to {IVFFLAT_PROBES}")
        elif INDEX_TYPE == "hnsw":
            cursor.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH};")
            logger.info(f"Set hnsw.ef_search to {HNSW_EF_SEARCH}")
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
    if index_type in ["hnsw", "ivfflat"]:
        return f"""
        SELECT file_name, document_page, chunk_no, chunk_text,
                (chunk_vector::halfvec(3072) <#> %s::halfvec(3072)) AS distance
        FROM document_vectors
        ORDER BY distance ASC
        LIMIT %s;
        """
    else:  # "none" or any other value
        return f"""
        SELECT file_name, document_page, chunk_no, chunk_text,
                (chunk_vector <#> %s::vector(3072)) AS distance
        FROM document_vectors
        ORDER BY distance ASC
        LIMIT %s;
        """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection established")
    try:
        while True:
            try:
                data = await websocket.receive_json()
                question = data["question"]
                top_n = data.get("top_n", 3)

                logger.info(f"Processing question: '{question}' with top_n={top_n}")

                question_vector = client.embeddings.create(
                    input=question,
                    model="text-embedding-3-large"
                ).data[0].embedding

                start_time = time.time()
                memory_usage = []

                with get_db_connection() as (conn, cursor):
                    search_query = get_search_query(INDEX_TYPE)
                    cursor.execute(search_query, (question_vector, top_n))
                    results = cursor.fetchall()

                    # pgvector_dbコンテナのメモリ使用量を記録
                    memory_usage.append(get_pgvector_db_memory_usage())

                end_time = time.time()
                search_time = end_time - start_time

                logger.info(f"Query returned {len(results)} results")

                formatted_results = []
                for result in results:
                    file_name, document_page, chunk_no, chunk_text, distance = result
                    formatted_result = {
                        "file_name": file_name,
                        "page": document_page,
                        "chunk_no": chunk_no,
                        "chunk_text": chunk_text,
                        "distance": float(distance),
                        "link_text": f"{file_name}, p.{document_page}",
                        "link": f"/pdf/{file_name}?page={document_page}",
                    }
                    formatted_results.append(formatted_result)

                    # pgvector_dbコンテナのメモリ使用量を記録
                    memory_usage.append(get_pgvector_db_memory_usage())

                avg_memory_usage = statistics.mean([m for m in memory_usage if m is not None])

                response_data = {
                    "results": formatted_results,
                    "search_time": search_time,
                    "avg_memory_usage": avg_memory_usage
                }

                await websocket.send_json(response_data)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                await websocket.send_json({"error": str(e)})
    finally:
        logger.info("WebSocket connection closed")

@app.get("/pdf/{file_name}")
async def get_pdf(file_name: str, page: int = None):
    pdf_path = os.path.join(PDF_INPUT_DIR, file_name)

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {file_name}")

    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        if page is not None:
            page = int(page)
            if 0 <= page < len(reader.pages):
                writer.add_page(reader.pages[page])
            else:
                raise HTTPException(status_code=400, detail=f"Invalid page number: {page}")
        else:
            # ページが指定されていない場合は全ページを含める
            for page in reader.pages:
                writer.add_page(page)

        buffer = BytesIO()
        writer.write(buffer)
        buffer.seek(0)

        headers = {
            "Content-Disposition": f"inline; filename={file_name}"
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
