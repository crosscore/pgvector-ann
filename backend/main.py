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
import psutil
import time
import statistics
from contextlib import contextmanager
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

def get_system_memory_usage():
    return psutil.virtual_memory().used / 1024  # KB

def get_postgresql_memory_usage(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sum(pg_total_relation_size(c.oid))::float
            FROM pg_class c
            LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE relkind = 'r' AND nspname NOT IN ('pg_catalog', 'information_schema')
        """)
        total_size = cur.fetchone()[0]

        cur.execute("SELECT pg_database_size(current_database())::float")
        db_size = cur.fetchone()[0]

        cur.execute("SELECT sum(pg_total_relation_size(relid))::float FROM pg_stat_user_tables")
        table_size = cur.fetchone()[0]

    return {
        'total_size': total_size / (1024) if total_size else 0,  # KB
        'db_size': db_size / (1024) if db_size else 0,  # KB
        'table_size': table_size / (1024) if table_size else 0  # KB
    }

class MemoryChangeContext:
    def __init__(self, conn):
        self.conn = conn
        self.memory_change = None

    def __enter__(self):
        self.initial_system_memory = get_system_memory_usage()
        self.initial_pg_memory = get_postgresql_memory_usage(self.conn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_system_memory = get_system_memory_usage()
        final_pg_memory = get_postgresql_memory_usage(self.conn)

        self.memory_change = {
            'system': final_system_memory - self.initial_system_memory,
            'pg_total': final_pg_memory['total_size'] - self.initial_pg_memory['total_size'],
            'pg_db': final_pg_memory['db_size'] - self.initial_pg_memory['db_size'],
            'pg_tables': final_pg_memory['table_size'] - self.initial_pg_memory['table_size']
        }

def measure_memory_change(conn):
    return MemoryChangeContext(conn)

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

                with get_db_connection() as (conn, cursor):
                    with measure_memory_change(conn) as mc:
                        search_query = get_search_query(INDEX_TYPE)
                        cursor.execute(search_query, (question_vector, top_n))
                        results = cursor.fetchall()

                end_time = time.time()
                search_time = end_time - start_time

                logger.info(f"Query returned {len(results)} results")
                logger.info(f"Search time: {search_time:.4f} seconds")

                if mc.memory_change:
                    logger.info(f"System memory change: {mc.memory_change['system']:.2f} KB")
                    logger.info(f"PostgreSQL total memory change: {mc.memory_change['pg_total']:.2f} KB")
                    logger.info(f"PostgreSQL DB memory change: {mc.memory_change['pg_db']:.2f} KB")
                    logger.info(f"PostgreSQL tables memory change: {mc.memory_change['pg_tables']:.2f} KB")
                else:
                    logger.warning("Unable to measure memory change")

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

                response_data = {
                    "results": formatted_results,
                    "search_time": search_time,
                    "memory_change": mc.memory_change
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
